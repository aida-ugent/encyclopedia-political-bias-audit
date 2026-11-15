"""
Batch Job Manager for persisting and retrieving OpenAI batch job state.

This module enables async workflows where batches are submitted, tracked,
and retrieved across separate process invocations.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class BatchJobMetadata:
    """Metadata for a submitted batch job."""
    batch_id: str
    model_name: str
    experiment_name: str
    run_id: str
    submitted_at: str
    status: str  # 'submitted', 'completed', 'failed', 'retrieved'
    request_count: int
    request_mapping: Dict[str, Dict[str, Any]]  # custom_id -> {row_index, source, name, ...}
    provider: str = "openai"  # 'openai' or 'anthropic'
    output_file: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class BatchJobManager:
    """
    Manages persistence and retrieval of batch job metadata.

    Stores job state in JSON files so batches can be tracked across process restarts.

    Directory structure:
        results/batch_jobs/
            {experiment}/
                {provider}_{model}/
                    {batch_id}/
                        metadata.json
                        batch_raw_results.csv
                        results.csv
    """

    def __init__(self, base_dir: str = "results/batch_jobs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for use in directory names."""
        # Replace problematic characters with underscores
        return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")

    def _job_dir_from_job(self, job: BatchJobMetadata) -> Path:
        """Get the directory for a batch job using job metadata."""
        model_dir_name = f"{job.provider}_{self._sanitize_model_name(job.model_name)}"
        # Include batch_id to prevent multiple batches from overwriting each other
        job_dir = self.base_dir / job.experiment_name / model_dir_name / job.batch_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _find_job_dir(self, batch_id: str) -> Optional[Path]:
        """Find the directory for a batch job by searching for batch_id in metadata."""
        for experiment_dir in self.base_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
            # Check each provider/model subdirectory
            for model_dir in experiment_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                # Check each batch_id subdirectory
                for batch_dir in model_dir.iterdir():
                    if not batch_dir.is_dir():
                        continue
                    metadata_file = batch_dir / "metadata.json"
                    if not metadata_file.exists():
                        continue
                    try:
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                        if data.get("batch_id") == batch_id:
                            return batch_dir
                    except Exception:
                        continue
        return None

    def save_job(self, job: BatchJobMetadata) -> str:
        """Save job metadata to disk. Returns the file path."""
        job_dir = self._job_dir_from_job(job)
        path = job_dir / "metadata.json"
        with open(path, 'w') as f:
            json.dump(asdict(job), f, indent=2)
        return str(path)

    def load_job(self, batch_id: str) -> Optional[BatchJobMetadata]:
        """Load job metadata from disk by batch_id."""
        job_dir = self._find_job_dir(batch_id)
        if job_dir is None:
            return None
        metadata_file = job_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        return BatchJobMetadata(**data)

    def list_jobs(self, run_id: str = None, status: str = None, experiment: str = None) -> List[BatchJobMetadata]:
        """List all jobs, optionally filtered by run_id, status, and/or experiment."""
        jobs = []
        # Get experiment directories
        if experiment:
            experiment_dirs = [self.base_dir / experiment] if (self.base_dir / experiment).exists() else []
        else:
            experiment_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]

        for experiment_dir in experiment_dirs:
            # Check each provider_model subdirectory
            for model_dir in experiment_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                # Check each batch_id subdirectory
                for batch_dir in model_dir.iterdir():
                    if not batch_dir.is_dir():
                        continue
                    metadata_file = batch_dir / "metadata.json"
                    if not metadata_file.exists():
                        continue
                    try:
                        with open(metadata_file, 'r') as f:
                            data = json.load(f)
                        job = BatchJobMetadata(**data)
                        if run_id is not None and job.run_id != run_id:
                            continue
                        if status is None or job.status == status:
                            jobs.append(job)
                    except Exception as e:
                        print(f"Warning: Could not load {metadata_file}: {e}")

        return jobs

    def list_pending_jobs(self) -> List[BatchJobMetadata]:
        """List all jobs that are submitted but not yet retrieved."""
        return self.list_jobs(status="submitted") + self.list_jobs(status="completed")

    def update_status(self, batch_id: str, status: str,
                      error: str = None, output_file: str = None) -> None:
        """Update the status of a job."""
        job = self.load_job(batch_id)
        if job:
            job.status = status
            if error:
                job.error = error
            if output_file:
                job.output_file = output_file
            if status in ("completed", "failed", "retrieved"):
                job.completed_at = datetime.now().isoformat()
            self.save_job(job)

    def get_results_dir(self, batch_id: str) -> Optional[Path]:
        """Get the directory where results should be saved."""
        return self._find_job_dir(batch_id)


def create_request_mapping(
    df,
    sources: List[str],
    name_column: str = "Name",
) -> Dict[str, Dict[str, Any]]:
    """
    Create a mapping from custom_id to row metadata for batch result processing.

    Args:
        df: DataFrame being processed
        sources: List of source columns (e.g., ["Wikipedia_Text", "Grokipedia_Text"])
        name_column: Column containing the entity name

    Returns:
        Dict mapping custom_id to metadata dict
    """
    mapping = {}
    idx = 0

    for row_idx, row in df.iterrows():
        for source in sources:
            custom_id = f"req_{idx}"
            mapping[custom_id] = {
                "row_index": row_idx,
                "request_index": idx,
                "name": row[name_column],
                "source": source,
            }
            idx += 1

    return mapping
