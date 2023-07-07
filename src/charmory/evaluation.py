"""Armory Experiment Configuration Classes"""

# TODO: review the Optionals with @woodall

from armory.data.datasets import ArmoryDataGenerator
from art.estimators import BaseEstimator
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

MethodName = Callable[
    ..., Any
]  # reference to a python method e.g. "armory.attacks.weakest"
StrDict = Dict[str, Any]  # dictionary of string keys and any values


@dataclass
class Attack:
    function: MethodName
    kwargs: StrDict
    knowledge: Literal["white", "black"]
    use_label: bool = False
    type: Optional[str] = None


@dataclass
class DatasetConfig:
    name: str
    load_test_dataset: Callable[[], ArmoryDataGenerator]
    load_train_dataset: Optional[Callable[[], ArmoryDataGenerator]] = None


@dataclass
class Defense:
    function: MethodName
    kwargs: StrDict
    type: Literal[
        "Preprocessor",
        "Postprocessor",
        "Trainer",
        "Transformer",
        "PoisonFilteringDefense",
    ]


@dataclass
class Metric:
    profiler_type: Literal["basic", "deterministic"]
    supported_metrics: List[str]
    perturbation: List[str]
    task: List[str]
    means: bool
    record_metric_per_sample: bool


@dataclass
class ModelConfig:
    name: str
    load_model: Callable[[], BaseEstimator]
    fit: bool = False
    fit_kwargs: StrDict = field(default_factory=dict)
    predict_kwargs: StrDict = field(default_factory=dict)


@dataclass
class Scenario:
    function: MethodName
    kwargs: StrDict


@dataclass
class SysConfig:
    # TODO: should get ArmoryControls (e.g. num_eval_batches, num_epochs, etc.)
    gpus: List[str]
    use_gpu: bool = False


@dataclass
class Evaluation:
    name: str
    description: str
    author: Optional[str]
    model: ModelConfig
    scenario: Scenario
    dataset: DatasetConfig
    attack: Optional[Attack] = None
    defense: Optional[Defense] = None
    metric: Optional[Metric] = None
    sysconfig: Optional[SysConfig] = None

    def asdict(self) -> dict:
        return asdict(self)

    def flatten(self):
        """return all parameters as (dot.path, value) pairs for externalization"""

        def flatten_dict(root, path):
            for key, value in root.items():
                if isinstance(value, dict):
                    yield from flatten_dict(value, path + [key])
                else:
                    yield ".".join(path + [key]), value

        return [x for x in flatten_dict(self.asdict(), [])]


# List of old armory environmental variables used in evaluations
# self.config.update({
#   "ARMORY_GITHUB_TOKEN": os.getenv("ARMORY_GITHUB_TOKEN", default=""),
#   "ARMORY_PRIVATE_S3_ID": os.getenv("ARMORY_PRIVATE_S3_ID", default=""),
#   "ARMORY_PRIVATE_S3_KEY": os.getenv("ARMORY_PRIVATE_S3_KEY", default=""),
#   "ARMORY_INCLUDE_SUBMISSION_BUCKETS": os.getenv(
#     "ARMORY_INCLUDE_SUBMISSION_BUCKETS", default=""
#   ),
#   "VERIFY_SSL": self.armory_global_config["verify_ssl"] or False,
#   "NVIDIA_VISIBLE_DEVICES": self.config["sysconfig"].get("gpus", None),
#   "PYTHONHASHSEED": self.config["sysconfig"].get("set_pythonhashseed", "0"),
#   "TORCH_HOME": paths.HostPaths().pytorch_dir,
#   environment.ARMORY_VERSION: armory.__version__,
#   # "HOME": "/tmp",
# })
