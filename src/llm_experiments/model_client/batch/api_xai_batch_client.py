"""
xAI Batch API client.

Asynchronous batch processing for xAI's Grok API.
https://docs.x.ai/developers/advanced-api-usage/batch-api

Workflow:
    1. Add requests with add_request()
    2. Submit with submit_batch() - returns batch_id immediately
    3. Check status with get_batch_status()
    4. Download results with download_results() when completed
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from .base_batch_client import BaseBatchClient, BatchRequest
from dotenv import load_dotenv

try:
    from xai_sdk import Client
    from xai_sdk.chat import system, user
    XAI_SDK_AVAILABLE = True
except ImportError:
    XAI_SDK_AVAILABLE = False


class ApiXAIBatchClient(BaseBatchClient):
    """
    Asynchronous xAI Batch API client.

    Usage:
        client = ApiXAIBatchClient(model_name="grok-4-1-fast-reasoning")  # or grok-3, grok-3-mini

        client.add_request("req_1", prompt="What is 2+2?")
        client.add_request("req_2", prompt="Capital of France?")

        batch_id = client.submit_batch()  # Returns immediately

        # Later, check and download when ready
        raw_path, parsed_path = client.download_results(batch_id, output_dir)

    Note:
        Requires the xai_sdk package: pip install xai-sdk
        Set XAI_API_KEY in environment or .env file.
        Batch API (Chat/GetCompletion) supports e.g. grok-4-1-fast-reasoning, grok-3, grok-3-mini;
        grok-4-0709 is not supported for this endpoint.
    """

    DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results"
    # Max number of requests per batch; larger sets are split into multiple batches.
    MAX_REQUESTS_PER_BATCH = 400
    # Max number of requests to send in a single batch.add() call; keep small to avoid gRPC
    # message size limits when scheduling jobs.
    MAX_REQUESTS_PER_ADD = 50

    def __init__(
        self,
        model_name: str,
        results_dir: Optional[str] = None,
        batch_name: Optional[str] = None,
    ):
        super().__init__(model_name)

        load_dotenv()
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not found in environment")

        # xai_sdk Client uses XAI_API_KEY from environment automatically
        self.client = Client(api_key=api_key)
        self.results_dir = Path(results_dir) if results_dir else self.DEFAULT_RESULTS_DIR
        self.batch_name = batch_name or "encyclopedia_analysis_batch"

    @property
    def provider_name(self) -> str:
        return "xai"

    def add_request(
        self,
        custom_id: str,
        prompt: str = None,
        message: List[Dict[str, str]] = None,
        max_new_tokens: int = 1,
    ) -> None:
        """Add a request to the pending batch."""
        if prompt is None and message is None:
            raise ValueError("Either prompt or message must be provided")

        self._pending_requests.append(BatchRequest(
            custom_id=custom_id,
            prompt=prompt,
            message=message,
            max_new_tokens=max_new_tokens,
        ))

    def _build_batch_requests(self, requests: List[BatchRequest]) -> List:
        """Convert BatchRequest objects to xAI's Chat format."""
        batch_requests = []

        for req in requests:
            # Create a Chat object for each request
            chat = self.client.chat.create(
                model=self.model_name,
                batch_request_id=req.custom_id,
            )

            if req.message:
                # Convert message list to Chat format
                for msg in req.message:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        chat.append(system(content))
                    else:
                        chat.append(user(content))
            else:
                # Simple prompt
                chat.append(user(req.prompt))

            batch_requests.append(chat)

        return batch_requests

    def _submit_single_batch(self, requests: List[BatchRequest], batch_index: int = 0, total_batches: int = 1) -> str:
        """
        Submit a single batch to xAI's Batch API.

        Args:
            requests: The requests to include in this batch (max MAX_REQUESTS_PER_BATCH).
            batch_index: Index of this batch within the overall split (for logging).
            total_batches: Total number of batches being submitted (for logging).

        Returns:
            The batch_id for the created batch.
        """
        batch_label = f"{self.batch_name}_{batch_index+1}of{total_batches}" if total_batches > 1 else self.batch_name
        batch = self.client.batch.create(batch_name=batch_label)
        batch_id = batch.batch_id

        batch_requests = self._build_batch_requests(requests)
        for i in range(0, len(batch_requests), self.MAX_REQUESTS_PER_ADD):
            chunk = batch_requests[i:i + self.MAX_REQUESTS_PER_ADD]
            # Retry batch.add up to 3 times in case of transient errors
            for attempt in range(3):
                try:
                    self.client.batch.add(batch_id=batch_id, batch_requests=chunk)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    # Small backoff before retrying
                    time.sleep(2)
            # Space out add calls slightly even on success
            time.sleep(2)

        if total_batches > 1:
            print(f"Batch {batch_index+1}/{total_batches} submitted: {batch_id} ({len(requests)} requests)")
        else:
            print(f"Batch submitted: {batch_id} ({len(requests)} requests)")
        return batch_id

    def submit_batch(self, requests: List[BatchRequest] = None) -> List[str]:
        """
        Submit requests to xAI's Batch API, splitting into multiple batches if needed.

        Batches are capped at MAX_REQUESTS_PER_BATCH requests each. When multiple
        batches are created, a 1-second delay is inserted between creations.

        Returns a list of batch_ids (one per sub-batch).
        """
        if requests is None:
            requests = self._pending_requests
            self._pending_requests = []

        if not requests:
            raise ValueError("No requests to submit")

        # Split requests into sub-batches of at most MAX_REQUESTS_PER_BATCH
        chunks = [
            requests[i:i + self.MAX_REQUESTS_PER_BATCH]
            for i in range(0, len(requests), self.MAX_REQUESTS_PER_BATCH)
        ]
        total_batches = len(chunks)

        batch_ids = []
        for idx, chunk in enumerate(chunks):
            if idx > 0:
                time.sleep(2)  #interval between batch creations
            batch_id = self._submit_single_batch(chunk, batch_index=idx, total_batches=total_batches)
            batch_ids.append(batch_id)

        print(f"Total: {len(batch_ids)} batch(es) submitted with {len(requests)} requests")
        return batch_ids

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch.

        Returns:
            Dict with:
            - id: Batch ID
            - status: 'in_progress', 'completed', 'failed', 'cancelled'
            - request_counts: Dict with total, completed, failed
            - created_at, expires_at: Timestamps
        """
        batch = self.client.batch.get(batch_id=batch_id)

        state = batch.state
        pending = state.num_pending
        total = state.num_requests
        success = state.num_success
        error = state.num_error
        cancelled = state.num_cancelled
        if pending == 0:
            if error > 0 and success == 0:
                status = "failed"
            elif cancelled == total:
                status = "cancelled"
            else:
                status = "completed"
        else:
            status = "in_progress"

        return {
            "id": batch_id,
            "status": status,
            "request_counts": {
                "total": total,
                "completed": success,
                "failed": error,
                "pending": pending,
                "cancelled": cancelled,
            },
            "created_at": getattr(batch, 'created_at', None),
            "expires_at": getattr(batch, 'expires_at', None),
        }

    def download_results(
        self,
        batch_id: str,
        output_dir: Path,
    ) -> Optional[Tuple[Path, Path]]:
        """
        Download and store results from a completed batch.

        Saves two files:
        - batch_raw_results.csv: Raw results with custom_id, content, status, error
        - results.csv: Parsed JSON content with custom_id, Name, Source, and extracted fields

        Args:
            batch_id: The batch ID to download results for
            output_dir: Directory where results should be saved

        Returns:
            Tuple of (raw_results_path, parsed_results_path), or None if batch not completed
        """
        status = self.get_batch_status(batch_id)

        if status["status"] not in ("completed", "failed"):
            print(f"Batch {batch_id} not completed yet (status: {status['status']})")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_results_path = output_dir / "batch_raw_results.csv"
        parsed_results_path = output_dir / "results.csv"

        # Load metadata for Name/Source mapping
        request_mapping = self._load_metadata(output_dir)

        # Paginate through all results
        all_succeeded = []
        all_failed = []
        pagination_token = None

        while True:
            page = self.client.batch.list_batch_results(
                batch_id=batch_id,
                limit=100,
                pagination_token=pagination_token,
            )

            all_succeeded.extend(page.succeeded)
            all_failed.extend(page.failed)

            if page.pagination_token is None:
                break
            pagination_token = page.pagination_token

        # Collect all results for processing
        raw_rows = []
        parsed_rows = []
        all_json_keys = set()

        for result in all_succeeded:
            content = result.response.content if hasattr(result.response, 'content') else ""
            raw_rows.append({
                "custom_id": result.batch_request_id,
                "content": content,
                "status": "success",
                "error": "",
            })
            # Parse JSON content for the parsed results file
            row_data, keys = self._build_parsed_row(result.batch_request_id, content, request_mapping)
            if row_data:
                all_json_keys.update(keys)
                parsed_rows.append(row_data)

        for result in all_failed:
            raw_rows.append({
                "custom_id": result.batch_request_id,
                "content": "",
                "status": "error",
                "error": result.error_message,
            })

        # Write results to CSV files
        self._write_raw_results_csv(raw_rows, raw_results_path)
        self._write_parsed_results_csv(parsed_rows, all_json_keys, parsed_results_path)

        print(f"Raw results saved to: {raw_results_path}")
        print(f"Parsed results saved to: {parsed_results_path}")
        return raw_results_path, parsed_results_path

    def cancel_batch(self, batch_id: str) -> str:
        """Cancel a batch. Returns the batch_id of cancelled batch."""
        cancelled_batch = self.client.batch.cancel(batch_id=batch_id)
        return cancelled_batch.batch_id

    def list_batches(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List recent batches.

        Returns:
            List of batch info dicts
        """
        response = self.client.batch.list(limit=limit)

        batches = []
        for batch in response.batches:
            status = "complete" if batch.state.num_pending == 0 else "processing"
            batches.append({
                "id": batch.batch_id,
                "name": batch.name,
                "status": status,
                "total": batch.state.num_requests,
                "completed": batch.state.num_success,
                "failed": batch.state.num_error,
            })

        return batches

    def get_max_position_embeddings(self) -> int:
        return 128000  # GPT-4o context window
