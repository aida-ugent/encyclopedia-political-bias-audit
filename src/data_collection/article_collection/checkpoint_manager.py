import json
from pathlib import Path
from config import logger


class CheckpointManager:
    """Manages checkpoints for resumable data collection."""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(checkpoint_file)
        self.processed_titles: set = set()
        self.load()

    def load(self):
        """Load checkpoint from file if it exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.processed_titles = set(data.get("processed_titles", []))
                logger.info(f"Loaded checkpoint with {len(self.processed_titles)} processed titles")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                self.processed_titles = set()

    def save(self):
        """Save checkpoint to file."""
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump({"processed_titles": list(self.processed_titles)}, f)

    def mark_processed(self, title: str):
        """Mark a title as processed."""
        self.processed_titles.add(title)

    def is_processed(self, title: str) -> bool:
        """Check if a title has been processed."""
        return title in self.processed_titles

    def clear(self):
        """Clear checkpoint data."""
        self.processed_titles = set()
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
