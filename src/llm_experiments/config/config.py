"""Centralized configuration for LLM experiments.

All hyperparameters are defined here for easy modification and WANDB logging.
"""
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

from experiments.ideology import Ideology

from .models import LOCAL_MODELS, API_MODELS, BATCH_MODELS


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    # Experiment settings
    experiments: List[Ideology] = field(default_factory=lambda: [Ideology()])
    dataset: str = "articles/LeadersAndParties/leaders_processed"
    # dataset: str = "100_political_figures"

    # WANDB settings
    wandb_project: str = "encyclopedia-analysis"
    wandb_entity: Optional[str] = None  # Set to your wandb username/team
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: str = ""

    def get_local_models(self) -> Dict[str, str]:
        return LOCAL_MODELS.copy()

    def get_api_models(self) -> Dict[str, str]:
        return API_MODELS.copy()

    def get_batch_models(self) -> Dict[str, str]:
        return BATCH_MODELS.copy()

    def to_dict(self):
        """Convert config to dictionary for WANDB logging."""
        return asdict(self)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        """Create config from parsed command line arguments."""
        return cls(
            experiments=args.experiments,
            dataset=args.dataset,
            wandb_project=args.project,
            wandb_entity=args.entity,
            wandb_tags=args.tags if args.tags else [],
            wandb_notes=args.notes,
        )


# Default configuration instance
default_config = ExperimentConfig()
