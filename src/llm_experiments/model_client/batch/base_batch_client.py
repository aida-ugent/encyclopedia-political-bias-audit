"""
Abstract base class for batch API clients.

This provides a common interface for different providers' batch APIs:
- OpenAI Batch API
- Anthropic Message Batches API
- Google Vertex AI Batch Prediction
- etc.

Workflow (asynchronous, never waits):
    1. Add requests with add_request()
    2. Submit with submit_batch() - returns list of batch_ids immediately
       (may split into multiple batches if exceeding provider limits)
    3. Check status with get_batch_status()
    4. Download results with download_results() when completed
"""

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

@dataclass
class BatchRequest:
    """Provider-agnostic batch request."""
    custom_id: str
    prompt: str = None
    message: List[Dict[str, str]] = None
    max_new_tokens: int = 1


class BaseBatchClient(ABC):
    """
    Abstract base class for asynchronous batch API clients.

    All operations are non-blocking. Batches are submitted and tracked,
    but never waited on. Results are downloaded when ready.

    Required methods:
    - add_request(): Queue a request for batching
    - submit_batch(): Submit queued requests, returns batch_id immediately
    - get_batch_status(): Check batch status
    - download_results(): Download and save results when completed
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pending_requests: List[BatchRequest] = []

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass

    @abstractmethod
    def add_request(
        self,
        custom_id: str,
        prompt: str = None,
        message: List[Dict[str, str]] = None,
        max_new_tokens: int = 1,
    ) -> None:
        """Add a request to the pending batch."""
        pass

    @abstractmethod
    def submit_batch(self, requests: List[BatchRequest] = None) -> List[str]:
        """
        Submit batch(es) to the provider.

        May split requests into multiple batches if they exceed provider limits.
        Returns immediately with list of batch_ids. Does not wait for completion.
        """
        pass

    @abstractmethod
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the status of a batch.

        Returns:
            Dict with at minimum:
            - status: str ('in_progress', 'completed', 'failed', etc.)
            - request_counts: Dict with 'total', 'completed', 'failed'
        """
        pass

    @abstractmethod
    def download_results(self, batch_id: str, output_dir: Path) -> Optional[Tuple[Path, Path]]:
        """
        Download and save results from a completed batch.

        Saves two files:
        - batch_raw_results.csv: Raw results with custom_id, content, status, error
        - results.csv: Parsed JSON content with custom_id, Name, Source, and extracted fields

        Returns:
            Tuple of (raw_results_path, parsed_results_path), or None if batch not completed
        """
        pass

    # -------------------------------------------------------------------------
    # Shared helper methods (concrete implementations)
    # -------------------------------------------------------------------------

    def _parse_json_content(self, content: str) -> Dict[str, str]:
        """
        Parse JSON content from LLM response and extract fields.

        Handles various JSON formats:
        - Standard JSON
        - JSON with double-double quotes (""key"" -> "key")
        - JSON embedded in markdown code blocks

        Normalizes field names (e.g., 'neutrality assessment' -> 'neutrality_assessment').
        """
        if not content:
            return {}

        # Try to extract JSON from markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            content = "\n".join(lines)

        # Fix double-double quotes (""key"" -> "key")
        # This handles malformed JSON like: {""explanation"": ""value""}
        if '""' in content:
            content = content.replace('""', '"')

        try:
            parsed = json.loads(content)
            # Normalize keys: replace spaces with underscores
            return {k.replace(' ', '_'): v for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError, AttributeError):
            return {}

    def _load_metadata(self, output_dir: Path) -> Dict[str, Dict[str, str]]:
        """
        Load metadata.json and return request_mapping.

        Returns:
            Dict mapping custom_id to {'name': ..., 'source': ...}
        """
        metadata_path = Path(output_dir) / "metadata.json"
        if not metadata_path.exists():
            print(f"Warning: metadata.json not found at {metadata_path}")
            return {}

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get("request_mapping", {})
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load metadata.json: {e}")
            return {}

    def _write_raw_results_csv(
        self,
        raw_rows: List[Dict[str, str]],
        output_path: Path,
    ) -> None:
        """Write raw results to CSV file."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["custom_id", "content", "status", "error"])
            writer.writeheader()
            writer.writerows(raw_rows)

    def _write_parsed_results_csv(
        self,
        parsed_rows: List[Dict[str, Any]],
        all_json_keys: set,
        output_path: Path,
    ) -> None:
        """Write parsed results to CSV with Name and Source columns first."""
        parsed_fieldnames = ["custom_id", "Name", "Source"] + sorted(all_json_keys)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=parsed_fieldnames)
            writer.writeheader()
            writer.writerows(parsed_rows)

    def _build_parsed_row(
        self,
        custom_id: str,
        content: str,
        request_mapping: Dict[str, Dict[str, str]],
    ) -> Tuple[Optional[Dict[str, Any]], set]:
        """
        Parse content and build a row for the parsed results CSV.

        Returns:
            Tuple of (row_dict or None, set of json keys found)
        """
        parsed_content = self._parse_json_content(content)
        if not parsed_content:
            return None, set()

        row_data = {"custom_id": custom_id, **parsed_content}
        if custom_id in request_mapping:
            row_data["Name"] = request_mapping[custom_id].get("name", "")
            row_data["Source"] = request_mapping[custom_id].get("source", "")

        return row_data, set(parsed_content.keys())

    def clear_pending(self) -> None:
        """Clear all pending requests."""
        self._pending_requests = []

    def pending_count(self) -> int:
        """Return number of pending requests."""
        return len(self._pending_requests)

    def is_batch_client(self) -> bool:
        """Identify this as a batch-capable client."""
        return True


def create_batch_client(model_name: str, model_provider: str = None, **kwargs) -> BaseBatchClient:
    """
    Factory function to create a batch client for the specified provider.

    Args:
        model_name: Model identifier
        model_provider: Provider name ('openai', 'anthropic', 'xai')
        **kwargs: Additional arguments passed to the client constructor

    Returns:
        BaseBatchClient instance

    Example:
        client = create_batch_client("gpt-5-mini", "openai")
        client = create_batch_client("claude-3-opus", "anthropic")
        client = create_batch_client("grok-3", "xai")
    """
    # Import here to avoid circular dependencies
    if model_provider.lower() in ("anthropic", "claude", "api-claude-batch"):
        from .api_claude_batch_client import ApiClaudeBatchClient
        return ApiClaudeBatchClient(model_name=model_name, **kwargs)

    elif model_provider.lower() in ("xai", "grok", "api-xai-batch"):
        from .api_xai_batch_client import ApiXAIBatchClient
        return ApiXAIBatchClient(model_name=model_name, **kwargs)

    else:
        raise ValueError(f"Unknown batch provider: {model_provider}. "
                        f"Supported: anthropic, xai")
