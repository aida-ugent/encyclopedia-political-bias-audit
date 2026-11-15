"""
Batch processing utilities for LLM Batch APIs.

This package contains:
- job_manager: Persistence and tracking of batch jobs across process restarts
- api_openai_batch_client: OpenAI Batch API client
- api_claude_batch_client: Anthropic Message Batches API client
- api_xai_batch_client: xAI Batch API client
"""

from .job_manager import (
    BatchJobManager,
    BatchJobMetadata,
    create_request_mapping,
)

from .base_batch_client import (
    BaseBatchClient,
    BatchRequest,
    create_batch_client,
)

__all__ = [
    "BatchJobManager",
    "BatchJobMetadata",
    "create_request_mapping",
    "BaseBatchClient",
    "BatchRequest",
    "create_batch_client",
]
