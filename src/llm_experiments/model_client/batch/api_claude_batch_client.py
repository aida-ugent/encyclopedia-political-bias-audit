"""
Anthropic Message Batches API client.

Asynchronous batch processing for Anthropic's API.
https://docs.anthropic.com/en/docs/build-with-claude/message-batches

Workflow:
    1. Add requests with add_request()
    2. Submit with submit_batch() - returns batch_id immediately
    3. Check status with get_batch_status()
    4. Download results with download_results() when completed
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

from .base_batch_client import BaseBatchClient, BatchRequest
from dotenv import load_dotenv
import anthropic


class ApiClaudeBatchClient(BaseBatchClient):
    """
    Asynchronous Anthropic Message Batches API client.

    Usage:
        client = ApiClaudeBatchClient(model_name="claude-3-5-sonnet-20241022")

        client.add_request("req_1", prompt="What is 2+2?")
        client.add_request("req_2", prompt="Capital of France?")

        batch_id = client.submit_batch()  # Returns immediately

        # Later, check and download when ready
        raw_path, parsed_path = client.download_results(batch_id, output_dir)
    """

    DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results"

    def __init__(
        self,
        model_name: str,
        results_dir: Optional[str] = None,
    ):
        super().__init__(model_name)

        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.results_dir = Path(results_dir) if results_dir else self.DEFAULT_RESULTS_DIR

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def add_request(
        self,
        custom_id: str,
        prompt: str = None,
        message: List[Dict[str, str]] = None,
        max_new_tokens: int = 10000,
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

    def _build_batch_requests(self, requests: List[BatchRequest]) -> List[Dict]:
        """Convert BatchRequest objects to Anthropic's batch format."""
        return [
            {
                "custom_id": req.custom_id,
                "params": {
                    "model": self.model_name,
                    "max_tokens": req.max_new_tokens,
                    "messages": req.message or [{"role": "user", "content": req.prompt}],
                }
            }
            for req in requests
        ]

    def submit_batch(self, requests: List[BatchRequest] = None) -> List[str]:
        """
        Submit a batch to Anthropic's Message Batches API.

        Returns immediately with a list containing the batch_id. Does not wait for completion.
        """
        if requests is None:
            requests = self._pending_requests
            self._pending_requests = []

        if not requests:
            raise ValueError("No requests to submit")

        batch_requests = self._build_batch_requests(requests)
        batch = self.client.messages.batches.create(requests=batch_requests)

        print(f"Batch submitted: {batch.id} ({len(requests)} requests)")
        return [batch.id]

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch.

        Returns:
            Dict with:
            - id: Batch ID
            - status: 'in_progress', 'completed', 'cancelling', 'cancelled'
            - request_counts: Dict with total, completed, failed
            - created_at, ended_at: Timestamps
        """
        batch = self.client.messages.batches.retrieve(batch_id)

        status_map = {"ended": "completed", "canceling": "cancelling", "canceled": "cancelled"}

        return {
            "id": batch.id,
            "status": status_map.get(batch.processing_status, batch.processing_status),
            "request_counts": {
                "total": batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored,
                "completed": batch.request_counts.succeeded,
                "failed": batch.request_counts.errored,
            },
            "created_at": batch.created_at,
            "ended_at": batch.ended_at,
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

        if status["status"] != "completed":
            print(f"Batch {batch_id} not completed yet (status: {status['status']})")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_results_path = output_dir / "batch_raw_results.csv"
        parsed_results_path = output_dir / "results.csv"

        # Load metadata for Name/Source mapping
        request_mapping = self._load_metadata(output_dir)

        # Collect all results for processing
        raw_rows = []
        parsed_rows = []
        all_json_keys = set()

        for result in self.client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                content = "".join(
                    block.text for block in result.result.message.content
                    if hasattr(block, 'text')
                )
                raw_rows.append({
                    "custom_id": result.custom_id,
                    "content": content,
                    "status": "success",
                    "error": "",
                })
                # Parse JSON content for the parsed results file
                row_data, keys = self._build_parsed_row(result.custom_id, content, request_mapping)
                if row_data:
                    all_json_keys.update(keys)
                    parsed_rows.append(row_data)
            else:
                error = str(result.result.error) if hasattr(result.result, 'error') else "Unknown error"
                raw_rows.append({
                    "custom_id": result.custom_id,
                    "content": "",
                    "status": "error",
                    "error": error,
                })

        # Write results to CSV files
        self._write_raw_results_csv(raw_rows, raw_results_path)
        self._write_parsed_results_csv(parsed_rows, all_json_keys, parsed_results_path)

        print(f"Raw results saved to: {raw_results_path}")
        print(f"Parsed results saved to: {parsed_results_path}")
        return raw_results_path, parsed_results_path

    def cancel_batch(self, batch_id: str) -> str:
        """Cancel a batch. Returns the new status."""
        batch = self.client.messages.batches.cancel(batch_id)
        return batch.processing_status

    def list_batches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent batches."""
        batches = self.client.messages.batches.list(limit=limit)
        return [
            {
                "id": b.id,
                "status": b.processing_status,
                "request_counts": {
                    "total": b.request_counts.processing + b.request_counts.succeeded + b.request_counts.errored,
                    "completed": b.request_counts.succeeded,
                    "failed": b.request_counts.errored,
                },
                "created_at": b.created_at,
            }
            for b in batches.data
        ]

    def get_max_position_embeddings(self) -> int:
        return 200000
