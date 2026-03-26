import os
import argparse

tmp_root = os.environ.get("TMPDIR", "/tmp")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmp_root, "ecgdl_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tmp_root, "ecgdl_cache"))
os.environ.setdefault("ECGDL_USE_WANDB", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

from datasets.arr10000dataset import Arr10000Dataset
from datasets.cinc2017dataset import CincChallenge2017Dataset
from datasets.cpsc2018dataset import CPSC2018Dataset
from datasets.ptbxldataset import PTBXLDataset
from evaluation.experiment import Experiment
from models.acharya2017cnn import CNN
from models.cpsc2018winner import CPSCWinnerNet
from models.rtacnn import RTACNN
from models.transfer2020resnet import ResNet
from processing.slidingwindow import SlidingWindow
from processing.transform import Transform

DATASET_REGISTRY = {
    "arr10000": Arr10000Dataset,
    "cinc2017": CincChallenge2017Dataset,
    "cpsc2018": CPSC2018Dataset,
    "ptb-xl": PTBXLDataset,
}

MODEL_REGISTRY = {
    "rtacnn": (RTACNN, 30, 300),
    "resnet": (ResNet, 2.5, 250),
    "cpscwinner": (CPSCWinnerNet, 30, 100),
    "cnn": (CNN, 10, 360),
}

TASK_DATASETS = {
    "form": [("cpsc2018", 30, 60), ("ptb-xl", 10, 10), ("arr10000", 10, 10)],
    "rhythm": [("ptb-xl", 10, 10), ("arr10000", 10, 10)],
    "cinc2017": [("cpsc2018", 30, 60), ("cinc2017", 10, 30), ("ptb-xl", 10, 10), ("arr10000", 10, 10)],
    "cpsc2018": [("cpsc2018", 30, 60), ("ptb-xl", 10, 10)],
    "aami": [("cpsc2018", 30, 30), ("ptb-xl", 10, 10), ("arr10000", 10, 10)],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ECG classification experiments.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["aami"],
        choices=sorted(TASK_DATASETS.keys()),
        help="Task definitions to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Models to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Optional dataset filter.",
    )
    parser.add_argument(
        "--eval",
        default="inter",
        choices=["inter", "intra", "fixed"],
        help="Evaluation strategy.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("ECGDL_EPOCHS", "100")),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--list-config",
        action="store_true",
        help="Print the resolved experiment plan and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print experiments without executing them.",
    )
    return parser.parse_args()


def resolve_experiments(args):
    selected_datasets = set(args.datasets) if args.datasets else None
    resolved = []

    for model_name in args.models:
        model_cls, sec, freq = MODEL_REGISTRY[model_name]
        for task in args.tasks:
            for dataset_name, threshold, alternative_sec in TASK_DATASETS[task]:
                if selected_datasets and dataset_name not in selected_datasets:
                    continue
                dataset_cls = DATASET_REGISTRY[dataset_name]
                if sec >= threshold:
                    transform_cls = Transform
                    input_seconds = threshold
                else:
                    transform_cls = SlidingWindow
                    input_seconds = sec
                resolved.append(
                    {
                        "task": task,
                        "dataset_name": dataset_name,
                        "dataset_cls": dataset_cls,
                        "model_name": model_name,
                        "model_cls": model_cls,
                        "transform_cls": transform_cls,
                        "input_seconds": input_seconds,
                        "freq": freq,
                    }
                )
    return resolved


def main():
    args = parse_args()
    experiments = resolve_experiments(args)

    if not experiments:
        raise ValueError("No experiments matched the requested configuration.")

    for exp in experiments:
        print(
            "task={task} dataset={dataset} model={model} transform={transform} "
            "seconds={seconds} fs={freq} eval={eval} epochs={epochs}".format(
                task=exp["task"],
                dataset=exp["dataset_name"],
                model=exp["model_name"],
                transform=exp["transform_cls"].__name__,
                seconds=exp["input_seconds"],
                freq=exp["freq"],
                eval=args.eval,
                epochs=args.epochs,
            )
        )

    if args.list_config or args.dry_run:
        return

    for exp in experiments:
        experiment = Experiment(
            exp["dataset_cls"],
            exp["transform_cls"],
            exp["freq"],
            exp["input_seconds"],
            exp["model_cls"],
            exp["task"],
            args.eval,
            args.epochs,
        )
        experiment.run()
        experiment.evaluate()


if __name__ == "__main__":
    main()
