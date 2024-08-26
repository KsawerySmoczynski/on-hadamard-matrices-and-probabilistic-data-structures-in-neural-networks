from enum import Enum, auto
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

TB_LOGS_PATTERN = "*.tfevents.*"


class MetricMonitoringMode(Enum):
    MIN = auto()
    MAX = auto()


def load_event_accumulator(path: Path) -> EventAccumulator:
    ea = EventAccumulator(str(path), size_guidance={SCALARS: 0})
    return ea.Reload()


def get_metrics(ea: EventAccumulator, metrics_to_get: dict[str, MetricMonitoringMode]) -> pd.DataFrame:
    results: dict[str, float] = {}
    mnames = ea.Tags()["scalars"]
    if missing_metrics := set(metrics_to_get) - set(mnames):
        raise ValueError(f"Missing metrics {missing_metrics} from {ea.path} log file")

    for metric_name, metric_monitoring_mode in metrics_to_get.items():
        df = pd.DataFrame(ea.Scalars(metric_name), columns=["wall_time", "step", "value"])
        if metric_monitoring_mode == MetricMonitoringMode.MIN:
            results[metric_name] = df["value"].min()
        elif metric_monitoring_mode == MetricMonitoringMode.MAX:
            results[metric_name] = df["value"].max()
        else:
            raise ValueError("Unknown monitoring mode")

    results["num_params"] = int(ea.Scalars("num_params")[0].value)

    return results


LOGS_ROOT_DIR = Path("lightning_logs")

EXPERIMENTS_CONFIGURATION = {
    "MNISTIdentityModule_MNISTProvider": {"test_loss": MetricMonitoringMode.MIN},
    "MNISTClassificationModule_MNISTProvider": {"test_accuracy": MetricMonitoringMode.MAX, "test_loss": MetricMonitoringMode.MIN},
    "MNISTMappingModule_MNISTProvider": {"test_loss": MetricMonitoringMode.MIN},
    "MNISTEntityMappingModule_MNISTMappedEntitiesProvider": {"test_loss": MetricMonitoringMode.MIN},
}


for experiment, metrics_config in EXPERIMENTS_CONFIGURATION.items():
    results = []
    for log_file_path in (LOGS_ROOT_DIR / experiment).rglob(TB_LOGS_PATTERN):
        event_accumulator = load_event_accumulator(log_file_path)
        if "test_loss" not in event_accumulator.Tags()["scalars"]:
            continue
        experiment_data = get_metrics(event_accumulator, metrics_config)
        _, experiment, model, experiment_name, date, hour, version, log_file_name = log_file_path.parts
        experiment_name = experiment_name.replace("_seed", "_s")
        hidden_idx = 12 if "64" in experiment_name else 13
        date, hidden_size, num_hidden, seed = experiment_name[:8], int(experiment_name[10:hidden_idx]), experiment_name[hidden_idx + 1 : -4], int(experiment_name[-2:])
        experiment_data["experiment"] = experiment
        experiment_data["model"] = model
        experiment_data["experiment_name"] = experiment_name
        experiment_data["hidden_size"] = hidden_size
        experiment_data["num_hidden"] = num_hidden
        experiment_data["seed"] = seed
        results.append(experiment_data)
    results_path = LOGS_ROOT_DIR / experiment / "raw_results.csv"
    results = pd.DataFrame(results)
    results.to_csv(results_path, index=False)
