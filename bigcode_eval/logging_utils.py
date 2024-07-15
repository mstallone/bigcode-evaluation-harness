import copy
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
from packaging.version import Version
import json

logger = logging.getLogger(__name__)


def get_wandb_printer() -> Literal["Printer"]:
    """Returns a wandb printer instance for pretty stdout."""
    from wandb.sdk.lib.printer import get_printer
    from wandb.sdk.wandb_settings import Settings

    printer = get_printer(Settings()._jupyter)
    return printer


class WandbLogger:
    def __init__(self, **kwargs) -> None:
        """Attaches to wandb logger if already initialized. Otherwise, passes kwargs to wandb.init()
        Args:
            kwargs Optional[Any]: Arguments for configuration.
        Parse and log the results returned from evaluator.simple_evaluate() with:
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            wandb_logger.log_eval_samples(results["samples"])
        """
        try:
            import wandb

            assert Version(wandb.__version__) >= Version("0.13.6")
            if Version(wandb.__version__) < Version("0.13.6"):
                wandb.require("report-editing:v0")
        except Exception as e:
            logger.warning(
                "To use the wandb reporting functionality please install wandb>=0.13.6.\n"
                "To install the latest version of wandb run `pip install wandb --upgrade`\n"
                f"{e}"
            )

        self.wandb_args: Dict[str, Any] = kwargs

        self.step = self.wandb_args.pop("step", 0)
        self.step_key = self.wandb_args.pop("step_key", "train/step")
        logger.info(f"Logging evaluations at {self.step_key}: {self.step}")

        # initialize a W&B run
        if wandb.run is None:
            self.run = wandb.init(**self.wandb_args)
        else:
            self.run = wandb.run

        self.run.define_metric(self.step_key, hidden=True)
        self.run.define_metric("evaluation/*", step_metric=self.step_key, step_sync=True)

        self.printer = get_wandb_printer()
    
    def post_init(self, results: Dict[str, Any]) -> None:
        self.results: Dict[str, Any] = copy.deepcopy(results)
        self.task_names: List[str] = list(results.get("results", {}).keys())

    def log_eval_result(self) -> None:
        """Log evaluation results to W&B."""
        # Log config to wandb
        self.run.config.update({"bigcode_eval": self.results.get("config", {})}, allow_val_change=True) # TODO: Ehhh, shouldn't really do this

        self.wandb_results = self._sanitize_results_dict()
        
        # Log the evaluation metrics to wandb
        for key in self.wandb_results:
            self.run.define_metric(key, step_metric=self.step_key, step_sync=True)
        self.wandb_results[self.step_key] =  self.step

        self.run.log(self.wandb_results)

        # Log the results dict as json to W&B Artifacts
        self._log_results_as_artifact()
    
    def _sanitize_results_dict(self) -> Dict[str, Any]:
        """Sanitize the results dictionary."""
        _results = copy.deepcopy(self.results.get("results", dict()))

        tmp_results = copy.deepcopy(_results)
        for task_name, task_results in tmp_results.items():
            for metric_name, metric_value in task_results.items():
                _results[f"{task_name}/{metric_name}"] = metric_value
                _results[task_name].pop(metric_name)
        for task in self.task_names:
            _results.pop(task)

        return _results
    
    def _log_results_as_artifact(self) -> None:
        """Log results as JSON artifact to W&B."""
        dumped = json.dumps(
            self.results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        artifact = wandb.Artifact("results", type="eval_results")
        with artifact.new_file("results.json", mode="w", encoding="utf-8") as f:
            f.write(dumped)
        self.run.log_artifact(artifact, aliases=["latest", f"{self.step_key}_{self.step}"])

def _handle_non_serializable(o: Any) -> Union[int, str, list]:
    """Handle non-serializable objects by converting them to serializable types.
    Args:
        o (Any): The object to be handled.
    Returns:
        Union[int, str, list]: The converted object. If the object is of type np.int64 or np.int32,
            it will be converted to int. If the object is of type set, it will be converted
            to a list. Otherwise, it will be converted to str.
    """
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

