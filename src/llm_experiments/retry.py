import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from config.config import ExperimentConfig
from experiments.ideology import Ideology
from model_client import create_batch_client, BatchJobManager, BatchJobMetadata

# Set this to the run directory you want to retry.
DEFAULT_RUN_DIR = Path("results/20260313_112452_og2wic6b")

def collect_failed_ids(batch_dir: Path) -> List[str]:
    """Return custom_ids with transport errors for a single batch directory."""
    raw_path = batch_dir / "batch_raw_results.csv"
    if not raw_path.exists():
        print(f"Skipping {batch_dir} (no batch_raw_results.csv)")
        return []

    raw_df = pd.read_csv(raw_path)
    failed_ids = raw_df.loc[
        (raw_df["status"] == "error") & (raw_df["error"] == "transport error"),
        "custom_id",
    ].tolist()

    if not failed_ids:
        print(f"No transport errors in {batch_dir.name}")
    else:
        print(f"{len(failed_ids)} transport-error requests in {batch_dir.name}")
    return failed_ids


def retry_batches_for_run(run_dir: Path) -> None:
    """
    Collect all failed transport-error requests across all batches for a run,
    and submit new batch(es) containing only those requests.

    Expects layout:
        {run_dir}/batch/{experiment}/{provider}_{model}/{batch_id}/...
    """
    batch_root = run_dir / "batch"
    if not batch_root.exists():
        print(f"No batch directory found under {run_dir}")
        return

    # We assume a single experiment/model/provider for now, matching Ideology usage.
    # Walk all batch_id directories and collect failed ids + their metadata.
    all_failed_ids: List[str] = []
    meta_paths: List[Path] = []

    for experiment_dir in batch_root.iterdir():
        if not experiment_dir.is_dir():
            continue
        for model_dir in experiment_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for batch_dir in model_dir.iterdir():
                if not batch_dir.is_dir():
                    continue
                failed = collect_failed_ids(batch_dir)
                if not failed:
                    continue
                all_failed_ids.extend(failed)
                meta_paths.append(batch_dir / "metadata.json")

    if not all_failed_ids:
        print("No transport-error requests found in any batch.")
        return

    # Load metadata from the first batch that has it
    meta_data = None
    for mp in meta_paths:
        if mp.exists():
            with open(mp, "r") as f:
                meta_data = json.load(f)
            break
    if meta_data is None:
        print("No metadata.json found for any batch, cannot reconstruct prompts.")
        return

    request_mapping = meta_data["request_mapping"]

    # Load original dataset and experiment
    config = ExperimentConfig()
    df = pd.read_csv(f"data/{config.dataset}.csv")
    exp = Ideology()

    # Create a new batch client (same model/provider as before)
    model_name = meta_data["model_name"]
    provider = meta_data.get("provider", "xai")
    client = create_batch_client(model_name, provider)

    # Prepare job manager for saving metadata for the new retry batch(es)
    job_manager = BatchJobManager(base_dir=str(batch_root))
    # Derive run_id from directory name: results/YYYYMMDD_HHMMSS_<run_id>
    name_parts = run_dir.name.split("_")
    run_id = name_parts[-1] if name_parts else run_dir.name

    unique_failed_ids = sorted(set(all_failed_ids))
    print(f"Retrying {len(unique_failed_ids)} unique transport-error requests")

    new_request_mapping = {}
    for cid in unique_failed_ids:
        info = request_mapping[cid]
        source = info["source"]
        name = info["name"]

        text = df[df["name"] == name][source].iloc[0]
        prompt = exp.build_prompt(name, text, client.get_max_position_embeddings())
        client.add_request(custom_id=cid, prompt=prompt)
        new_request_mapping[cid] = info
    new_batch_ids = client.submit_batch()
    print("New batch id(s):", new_batch_ids)

    # Save metadata for each new retry batch so it is tracked like original jobs
    for batch_id in new_batch_ids:
        job = BatchJobMetadata(
            batch_id=batch_id,
            model_name=model_name,
            experiment_name=exp.name(),
            run_id=run_id,
            submitted_at=datetime.now().isoformat(),
            status="submitted",
            request_count=len(new_request_mapping),
            request_mapping=new_request_mapping,
            provider=provider,
        )
        job_path = job_manager.save_job(job)
        print(f"Saved retry job metadata to: {job_path}")


if __name__ == "__main__":
    retry_batches_for_run(DEFAULT_RUN_DIR)
