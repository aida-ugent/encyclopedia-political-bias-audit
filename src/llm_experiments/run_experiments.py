#!/usr/bin/env python
"""
Unified experiment runner for LLM evaluations.

Orchestrates local HuggingFace models, synchronous API models, and asynchronous
batch models with a single run_id.

Directory structure:
    results/{timestamp}_{run_id}/
        local/
            {experiment}/
                {model}/
                    results.csv
        api/
            {experiment}/
                {model}/
                    results.csv
        batch/
            {experiment}/
                {provider}_{model}/
                    {batch_id}/
                        metadata.json
                        batch_raw_results.csv
                        results.csv

Usage:
    # Run all configured models (Local + API + Batch)
    python run_experiments.py submit

    # Check status of batch jobs
    python run_experiments.py status --run-id <run_id>

    # Retrieve completed batch results
    python run_experiments.py retrieve --run-id <run_id>

Workflow:
1. Configure models in config/models.py (LOCAL_MODELS, API_MODELS, and/or BATCH_MODELS)
2. Run 'submit' - Local and API models run synchronously, batch models submit async
3. Check 'status' for pending batch jobs
4. Run 'retrieve' when batch jobs complete
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from config.config import ExperimentConfig
from config.wandb_manager import WandBManager

def generate_run_dir(run_id: str) -> Path:
    """Generate the base results directory for a run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"results/{timestamp}_{run_id}")


def run_local_models(
    config: ExperimentConfig,
    df,
    output_dir: Path,
    local_models: dict
) -> None:
    """Run local HuggingFace model experiments."""
    from model_client.hf_model_client import HFModelClient

    local_output_dir = output_dir / "local"

    for model_name, model_path in local_models.items():
        print(f"\n{'='*60}")
        print(f"Loading local model: {model_name} from {model_path}")
        print(f"{'='*60}")

        client = HFModelClient(model_name = model_name, model_path=model_path)

        try:
            for exp in config.experiments:
                print(f"Running experiment: {exp.name()}")

                # Prepare output path for incremental writes
                results_path = local_output_dir / exp.name() / model_name / "results.csv"
                results_path.parent.mkdir(parents=True, exist_ok=True)

                exp.evaluate(client, df, output_path=results_path)

                print(f"Results saved to: {results_path}")
        finally:
            client.close()


def run_api_models(
    config: ExperimentConfig,
    df,
    output_dir: Path
) -> None:
    """Run synchronous API model experiments."""
    from model_client import create_api_model_client

    api_models = config.get_api_models()
    if not api_models:
        print("No API models configured, skipping synchronous evaluation.")
        return

    api_output_dir = output_dir / "api"

    for model_name, provider in api_models.items():
        print(f"\n{'='*60}")
        print(f"Running API model: {model_name}")
        print(f"{'='*60}")

        client = create_api_model_client(model_name, provider)

        try:
            for exp in config.experiments:
                print(f"Running experiment: {exp.name()}")

                # Prepare output path for incremental writes
                results_path = api_output_dir / exp.name() / model_name / "results.csv"
                results_path.parent.mkdir(parents=True, exist_ok=True)

                exp.evaluate(client, df, output_path=results_path)

                print(f"Results saved to: {results_path}")
        finally:
            client.close()


def run_batch_models(
    config: ExperimentConfig,
    df,
    output_dir: Path,
    run_id
) -> list:
    """Submit asynchronous batch model experiments. Returns list of batch_ids."""
    from model_client import create_batch_client, BatchJobManager, BatchJobMetadata

    batch_models = config.get_batch_models()
    if not batch_models:
        print("No batch models configured, skipping batch submission.")
        return []

    # Use custom base_dir under the unified output directory
    batch_base_dir = output_dir / "batch"
    job_manager = BatchJobManager(base_dir=str(batch_base_dir))

    batch_ids = []

    for model_name, model_provider in batch_models.items():
        client = create_batch_client(model_name, model_provider)

        for exp in config.experiments:
            print(f"\n{'='*60}")
            print(f"Submitting batch: {exp.name()} with model {model_name}")
            print(f"{'='*60}")

            exp_batch_ids, request_mapping = exp.evaluate_batch(client, df)
            batch_ids.extend(exp_batch_ids)

            # Save job metadata for each batch
            for i, batch_id in enumerate(exp_batch_ids):
                job = BatchJobMetadata(
                    batch_id=batch_id,
                    model_name=model_name,
                    experiment_name=exp.name(),
                    run_id=run_id,
                    submitted_at=datetime.now().isoformat(),
                    status="submitted",
                    request_count=len(request_mapping),
                    request_mapping=request_mapping,
                    provider=model_provider,
                )

                job_path = job_manager.save_job(job)
                print(f"  Job metadata saved to: {job_path}")
                if len(exp_batch_ids) > 1:
                    print(f"  Batch ID ({i+1}/{len(exp_batch_ids)}): {batch_id}")
                else:
                    print(f"  Batch ID: {batch_id}")

    return batch_ids


def submit_experiments(
    config: ExperimentConfig,
    resume_dir: Optional[str] = None,
) -> str:
    """
    Submit all configured experiments.

    Local models run synchronously, API models run synchronously, batch models submit and return asynchronously.
    Returns the run_id.

    If `resume_dir` points to an existing results/{timestamp}_{run_id} directory,
    that directory is reused instead of creating a new one. Synchronous experiments
    skip work already written to their results.csv, so an interrupted run continues
    from where it stopped.
    """
    wandb_manager = WandBManager(config)

    if resume_dir:
        output_dir = Path(resume_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {output_dir}")
        # Directory is named "{timestamp}_{run_id}"; recover the trailing run_id.
        run_id = output_dir.name.rsplit("_", 1)[-1]
        wandb_manager.init_run(resume_run_id=run_id)
        print(f"\n{'='*60}")
        print(f"RESUMING RUN: {run_id}")
    else:
        # Initialize WandB and get a fresh run_id
        wandb_manager.init_run()
        run_id = wandb_manager.run_id
        # Generate unified output directory
        output_dir = generate_run_dir(run_id)
        print(f"\n{'='*60}")
        print(f"EXPERIMENT RUN: {run_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    local_models = config.get_local_models()
    api_models = config.get_api_models()
    batch_models = config.get_batch_models()

    print(f"\nLocal models: {list(local_models.keys()) if local_models else 'None'}")
    print(f"API models: {list(api_models.keys()) if api_models else 'None'}")
    print(f"Batch models: {list(batch_models.keys()) if batch_models else 'None'}")

    df = pd.read_csv(f"data/{config.dataset}.csv")


    try:
        # Submit async batch models
        batch_ids = []
        if batch_models:
            print(f"\n{'='*60}")
            print("Asynchronous Batch Models (Submit)")
            print(f"{'='*60}")
            batch_ids = run_batch_models(config, df, output_dir, run_id)

        # Run local HuggingFace models
        if local_models:
            print(f"\n{'='*60}")
            print("Local HuggingFace Models")
            print(f"{'='*60}")
            run_local_models(config, df, output_dir, local_models)

        # Run synchronous API models
        if api_models:
            print(f"\n{'='*60}")
            print("Synchronous API Models")
            print(f"{'='*60}")
            run_api_models(config, df, output_dir)



        wandb_manager.log_status("submitted" if batch_ids else "completed")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Run ID: {run_id}")
        print(f"Output directory: {output_dir}")

        if local_models:
            print(f"Local results: {output_dir / 'local'}")

        if api_models:
            print(f"API results: {output_dir / 'api'}")

        if batch_ids:
            print(f"Batch jobs submitted: {len(batch_ids)}")
            print(f"Batch results (when ready): {output_dir / 'batch'}")
            print(f"\nUse 'python run_experiments.py status --run-id {run_id}' to check progress")
            print(f"Use 'python run_experiments.py retrieve --run-id {run_id}' when complete")

        print(f"{'='*60}\n")

    except Exception as e:
        wandb_manager.log_status("failed", str(e))
        raise
    finally:
        wandb_manager.finish()

    return run_id


def check_status(run_id: Optional[str] = None, base_dir: Optional[str] = None) -> None:
    """Check status of pending batch jobs."""
    from model_client import create_batch_client, BatchJobManager

    # Collect jobs from appropriate job managers
    jobs = []
    job_managers = {}  # batch_id -> job_manager mapping for updates

    if base_dir:
        manager = BatchJobManager(base_dir=base_dir)
        if run_id:
            for job in manager.list_jobs(run_id=run_id):
                jobs.append(job)
                job_managers[job.batch_id] = manager
        else:
            for job in manager.list_pending_jobs():
                jobs.append(job)
                job_managers[job.batch_id] = manager
    elif run_id:
        manager = _find_job_manager(run_id)
        if manager is None:
            manager = BatchJobManager(base_dir="results/batch_jobs")
        for job in manager.list_jobs(run_id=run_id):
            jobs.append(job)
            job_managers[job.batch_id] = manager
    else:
        # Search ALL batch directories
        managers = _find_all_job_managers()
        if not managers:
            managers = [BatchJobManager(base_dir="results/batch_jobs")]
        for manager in managers:
            for job in manager.list_pending_jobs():
                jobs.append(job)
                job_managers[job.batch_id] = manager

    if not jobs:
        print("No pending batch jobs found.")
        return

    print(f"\n{'='*80}")
    print(f"{'Batch ID':<30} {'Provider':<10} {'Experiment':<15} {'Status':<15} {'Progress':<15}")
    print(f"{'='*80}")

    clients = {}

    for job in jobs:
        job_manager = job_managers[job.batch_id]
        provider = getattr(job, 'provider', 'openai')
        if provider not in clients:
            clients[provider] = create_batch_client(job.model_name, provider)
        client = clients[provider]

        try:
            status_info = client.get_batch_status(job.batch_id)
            api_status = status_info["status"]
            counts = status_info["request_counts"]
            progress = f"{counts['completed']}/{counts['total']}"

            # Update local status if changed
            if api_status == "completed" and job.status != "completed":
                job_manager.update_status(job.batch_id, "completed")
            elif api_status in ("failed", "expired", "cancelled"):
                job_manager.update_status(job.batch_id, "failed", error=api_status)

            print(f"{job.batch_id:<30} {provider:<10} {job.experiment_name:<15} {api_status:<15} {progress:<15}")
        except Exception as e:
            print(f"{job.batch_id:<30} {provider:<10} {job.experiment_name:<15} {'ERROR':<15} {str(e)[:15]}")

    print(f"{'='*80}\n")


def retrieve_results(run_id: Optional[str] = None, retrieve_all: bool = False) -> None:
    """Retrieve results from completed batch jobs."""
    from model_client import create_batch_client, BatchJobManager

    # Collect jobs from appropriate job managers
    jobs = []
    job_managers = {}  # batch_id -> job_manager mapping for updates

    if retrieve_all:
        # Search ALL batch directories
        managers = _find_all_job_managers()
        if not managers:
            # Fall back to legacy location
            managers = [BatchJobManager(base_dir="results/batch_jobs")]

        for manager in managers:
            for job in manager.list_pending_jobs():
                jobs.append(job)
                job_managers[job.batch_id] = manager
            for job in manager.list_jobs(status="completed"):
                if job.batch_id not in job_managers:  # Avoid duplicates
                    jobs.append(job)
                    job_managers[job.batch_id] = manager
    elif run_id:
        job_manager = _find_job_manager(run_id)
        if job_manager is None:
            job_manager = BatchJobManager(base_dir="results/batch_jobs")
        for job in job_manager.list_jobs(run_id=run_id):
            jobs.append(job)
            job_managers[job.batch_id] = job_manager
    else:
        print("Specify --run-id or --all")
        return

    if not jobs:
        print("No jobs found to retrieve.")
        return

    clients = {}

    for job in jobs:
        if job.status == "retrieved":
            print(f"Skipping {job.batch_id} - already retrieved")
            continue

        job_manager = job_managers[job.batch_id]
        provider = getattr(job, 'provider', 'openai')
        if provider not in clients:
            clients[provider] = create_batch_client(job.model_name, provider)
        client = clients[provider]

        try:
            # Sync status from API
            status_info = client.get_batch_status(job.batch_id)
            api_status = status_info["status"]
            counts = status_info["request_counts"]

            if api_status in ("failed", "expired", "cancelled"):
                print(f"Skipping {job.batch_id} - {api_status}")
                job_manager.update_status(job.batch_id, "failed", error=api_status)
                continue

            if api_status != "completed":
                progress = f"{counts['completed']}/{counts['total']}"
                print(f"Skipping {job.batch_id} - {api_status} ({progress})")
                continue

            print(f"\nRetrieving {job.batch_id} ({job.experiment_name})...")

            # Get output directory from job manager
            output_dir = job_manager.get_results_dir(job.batch_id)
            if output_dir is None:
                print("  Could not find job directory")
                continue

            result = client.download_results(job.batch_id, output_dir)

            if result is None:
                print("  Download failed")
                continue

            raw_results_path, parsed_results_path = result

            # Update job status
            job_manager.update_status(
                job.batch_id, "retrieved",
                output_file=str(parsed_results_path)
            )

            print(f"  Results saved to: {parsed_results_path}")

        except Exception as e:
            print(f"  Error retrieving {job.batch_id}: {e}")
            job_manager.update_status(job.batch_id, "failed", error=str(e))


def _find_job_manager(run_id: Optional[str] = None):
    """Find the BatchJobManager for a given run_id by searching result directories."""
    from model_client import BatchJobManager

    results_dir = Path("results")
    if not results_dir.exists():
        return None

    # Sort by name (descending) to get most recent first
    entries = sorted(results_dir.iterdir(), key=lambda x: x.name, reverse=True)

    for entry in entries:
        if not entry.is_dir():
            continue

        # Check if this directory matches the run_id pattern
        if run_id and not entry.name.endswith(f"_{run_id}"):
            continue

        batch_dir = entry / "batch"
        if batch_dir.exists():
            return BatchJobManager(base_dir=str(batch_dir))

    return None


def _find_all_job_managers():
    """Find all BatchJobManagers across all result directories."""
    from model_client import BatchJobManager

    results_dir = Path("results")
    if not results_dir.exists():
        return []

    managers = []
    for entry in results_dir.iterdir():
        if not entry.is_dir():
            continue

        batch_dir = entry / "batch"
        if batch_dir.exists():
            managers.append(BatchJobManager(base_dir=str(batch_dir)))

    return managers


def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for API and Batch models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit experiments (API sync + Batch async)")
    submit_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help="Resume an interrupted run by reusing an existing results/{timestamp}_{run_id} directory "
             "(synchronous experiments skip rows already written to results.csv)",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check batch job status")
    status_parser.add_argument("--run-id", type=str, help="Specific run to check")

    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve completed batch results")
    retrieve_parser.add_argument("--run-id", type=str, help="Specific run to retrieve")
    retrieve_parser.add_argument("--all", action="store_true", default=False, help="Retrieve all completed")

    args = parser.parse_args()

    if args.command == "submit":
        config = ExperimentConfig()
        submit_experiments(config, resume_dir=args.resume)

    elif args.command == "status":
        check_status(args.run_id)

    elif args.command == "retrieve":
        retrieve_results(args.run_id, args.all)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
