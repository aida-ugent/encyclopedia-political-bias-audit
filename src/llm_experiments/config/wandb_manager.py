"""WandB logging manager for LLM experiments."""
import os
import wandb
import pandas as pd
from typing import Dict, Any, Optional
from .config import ExperimentConfig


class WandBManager:
    """Manages all WandB logging operations for experiments."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize WandB manager and start a new run.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.run = None

    def build_wandb_config(self) -> Dict[str, Any]:
        """Build complete config dict for WANDB logging."""
        wandb_config = self.config.to_dict()

        # Add experiment-specific params
        for exp in self.config.experiments:
            wandb_config[f"experiment_{exp.name()}"] = exp.params()

        # Add model info
        wandb_config["local_models"] = list(self.config.get_local_models().keys())
        wandb_config["api_models"] = list(self.config.get_api_models().keys())
        wandb_config["batch_api_models"] = list(self.config.get_batch_models().keys())

        return wandb_config

    def init_run(self, resume_run_id: Optional[str] = None):
        """Initialize WandB run.

        Args:
            resume_run_id: If provided, resume the existing WandB run with this id
                instead of creating a new one.
        """
        wandb_config = self.build_wandb_config()

        init_kwargs = dict(
            mode=os.getenv("WANDB_MODE", "offline"),
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config=wandb_config,
            tags=self.config.wandb_tags + [exp.name() for exp in self.config.experiments],
            notes=self.config.wandb_notes,
            group=self.config.dataset,  # Group runs by dataset
        )
        if resume_run_id:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "allow"

        self.run = wandb.init(**init_kwargs)

        return self.run

    def log_results(self, experiment_name: str, model_name: str, results_df: pd.DataFrame):
        """
        Log experiment results to WandB.

        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model
            results_df: DataFrame containing results
        """
        # Log sample count
        wandb.log({f"{experiment_name}/{model_name}/num_samples": len(results_df)})

        # Log results table
        wandb.log({
            f"{experiment_name}/{model_name}/results": wandb.Table(dataframe=results_df)
        })

    def log_artifact(self, experiment_name: str, model_name: str, file_path: str):
        """
        Log results file as WandB artifact.

        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model
            file_path: Path to the results file
        """
        artifact = wandb.Artifact(
            name=f"{experiment_name}-{model_name}-results",
            type="results",
            description=f"Results for {experiment_name} experiment using {model_name}"
        )
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)

    def log_status(self, status: str, error: str = None):
        """
        Log run status.

        Args:
            status: Status string (e.g., "completed", "failed")
            error: Optional error message if status is "failed"
        """
        log_dict = {"status": status}
        if error:
            log_dict["error"] = error
        wandb.log(log_dict)

    def finish(self):
        """Finish the WandB run."""
        if self.run:
            wandb.finish()

    @property
    def run_id(self) -> str:
        """Get the current run ID."""
        return self.run.id if self.run else None
