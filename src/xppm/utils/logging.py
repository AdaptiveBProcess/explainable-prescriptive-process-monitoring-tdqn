from __future__ import annotations

import json
import logging
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def get_logger(name: str = "xppm") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_env_fingerprint() -> dict[str, Any]:
    """Capture environment fingerprint (Python, packages, GPU info)."""
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }
    
    # PyTorch info
    env["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        env["cuda_available"] = True
        env["cuda_version"] = torch.version.cuda or "unknown"
        env["cudnn_version"] = str(torch.backends.cudnn.version())
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        env["cuda_available"] = False
    
    return env


def start_run_metadata(
    stage: str,
    config_path: str | Path,
    config_hash: str,
    seed: int,
    deterministic: bool,
    data_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Initialize run metadata at start of a stage.
    
    Args:
        stage: Stage name (train, ope, xai, etc.)
        config_path: Path to config file
        config_hash: Hash of config
        seed: Random seed used
        deterministic: Whether deterministic mode was enabled
        data_fingerprint: Hash/fingerprint of data files
    
    Returns:
        Metadata dict (to be finalized with finalize_run_metadata)
    """
    from .io import get_git_commit
    
    git_info = get_git_commit()
    run_id = f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    metadata = {
        "run_id": run_id,
        "stage": stage,
        "started_at": datetime.now().isoformat(),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "data_fingerprint": data_fingerprint,
        "seed": seed,
        "deterministic": deterministic,
        "env": get_env_fingerprint(),
    }
    
    return metadata


def finalize_run_metadata(
    metadata: dict[str, Any],
    outputs: list[str | Path] | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize run metadata with completion time and outputs.
    
    Args:
        metadata: Metadata dict from start_run_metadata
        outputs: List of output file paths created
        metrics: Optional metrics dict to include
    
    Returns:
        Finalized metadata dict
    """
    metadata["finished_at"] = datetime.now().isoformat()
    if outputs:
        metadata["outputs"] = [str(p) for p in outputs]
    if metrics:
        metadata["metrics"] = metrics
    
    return metadata


def save_run_metadata(metadata: dict[str, Any], output_path: str | Path) -> None:
    """Save run metadata to JSON file."""
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# Tracking wrapper (MLflow/W&B)
class Tracker:
    """Unified tracking wrapper for MLflow or W&B."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize tracker from config.
        
        Args:
            config: Tracking config dict with 'enabled', 'backend', etc.
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.backend = config.get("backend", "mlflow")
        self.run = None
        self._mlflow_run = None
        self._wandb_run = None
        
        if not self.enabled:
            return
        
        if self.backend == "mlflow":
            try:
                import mlflow
                self.mlflow = mlflow
            except ImportError:
                get_logger(__name__).warning("MLflow not installed, tracking disabled")
                self.enabled = False
        elif self.backend == "wandb":
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                get_logger(__name__).warning("W&B not installed, tracking disabled")
                self.enabled = False
    
    def init_run(
        self,
        run_name: str,
        stage: str,
        tags: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new tracking run.
        
        Args:
            run_name: Name for this run
            stage: Stage name (train, ope, xai, etc.)
            tags: Optional tags dict
            params: Optional params dict (will be merged with metadata)
        """
        if not self.enabled:
            return
        
        logger = get_logger(__name__)
        
        if self.backend == "mlflow":
            mlflow_cfg = self.config.get("mlflow", {})
            self.mlflow.set_experiment(mlflow_cfg.get("experiment_name", "xppm-tdqn"))
            self._mlflow_run = self.mlflow.start_run(run_name=run_name)
            self.run = self._mlflow_run
            
            # Set tags
            all_tags = {"stage": stage}
            if tags:
                all_tags.update(tags)
            for key, value in all_tags.items():
                self.mlflow.set_tag(key, str(value))
            
            # Set params
            if params:
                for key, value in params.items():
                    self.mlflow.log_param(key, value)
            
            logger.info("MLflow run started: %s", run_name)
        
        elif self.backend == "wandb":
            wandb_cfg = self.config.get("wandb", {})
            self._wandb_run = self.wandb.init(
                project=wandb_cfg.get("project", "xppm-tdqn"),
                entity=wandb_cfg.get("entity"),
                name=run_name,
                tags=self.config.get("wandb", {}).get("tags", []) + [stage],
            )
            self.run = self._wandb_run
            
            # Set config (params)
            if params:
                self.wandb.config.update(params)
            
            logger.info("W&B run started: %s", run_name)
    
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to tracker.
        
        Args:
            metrics: Dict of metric name -> value
            step: Optional step/epoch number
        """
        if not self.enabled:
            return
        
        if self.backend == "mlflow":
            self.mlflow.log_metrics(metrics, step=step)
        elif self.backend == "wandb":
            self.wandb.log(metrics, step=step)
    
    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log artifact (file) to tracker.
        
        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifacts (for organization)
        """
        if not self.enabled:
            return
        
        path = Path(local_path)
        if not path.exists():
            get_logger(__name__).warning("Artifact not found: %s", local_path)
            return
        
        if self.backend == "mlflow":
            self.mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        elif self.backend == "wandb":
            self.wandb.log_artifact(str(local_path), name=artifact_path or path.name)
    
    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on the run.
        
        Args:
            tags: Dict of tag name -> value
        """
        if not self.enabled:
            return
        
        if self.backend == "mlflow":
            for key, value in tags.items():
                self.mlflow.set_tag(key, str(value))
        elif self.backend == "wandb":
            # W&B tags are a list, so we update config
            for key, value in tags.items():
                self.wandb.config[key] = value
    
    def finish(self) -> None:
        """Finish/close the tracking run."""
        if not self.enabled:
            return
        
        if self.backend == "mlflow" and self._mlflow_run:
            self.mlflow.end_run()
        elif self.backend == "wandb" and self._wandb_run:
            self.wandb.finish()


def init_tracker(config: dict[str, Any]) -> Tracker:
    """Initialize tracker from config.
    
    Args:
        config: Tracking config dict
    
    Returns:
        Tracker instance
    """
    return Tracker(config)


