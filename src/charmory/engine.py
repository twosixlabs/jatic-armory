import datetime
import os
import shutil

import armory
from armory import environment, paths
from armory.configuration import load_global_config
from armory.logs import log
from armory.scenarios.main import main as scenario_main
from armory.utils.printing import bold, red


class Engine:
    """
    Engine control launching of ARMORY evaluations.
    """
    def __init__(self, evaluation):
        self.evaluation = evaluation

        metadata = evaluation._metadata
        mlexp = mlflow.get_experiment_by_name(metadata.name)
        if mlexp:
            self.experiment_id = mlexp.experiment_id
            log.info(f"Experiment {metadata.name} already exists {self.experiment_id}")
        else:
            self.experiment_id = mlflow.create_experiment(
                metadata.name,
            )
            log.info(
                f"Creating experiment {self.evaluation._metadata.name} as {self.experiment_id}"
            )

    def run(self):
        log.info(bold(f"Running Evaluation{red(self.experiment._metadata.name)}"))
        result = scenario_main(self.evaluation)
        result["benign"] = id(self.experiment.model)
        if self.experiment.attack:
            result["attack"] = id(self.experiment.attack)
        return result


# class Evaluator:
#     def __init__(self, config: dict):
#         log.info("Constructing Evaluator Object")
#         if not isinstance(config, dict):
#             raise ValueError(f"config {config} must be a dict")
#         self.config = config

#         self.host_paths = paths.HostPaths()
#         if os.path.exists(self.host_paths.armory_config):
#             self.armory_global_config = load_global_config(
#                 self.host_paths.armory_config
#             )
#         else:
#             self.armory_global_config = {"verify_ssl": True}

#         # Output directory configuration
#         date_time = datetime.datetime.utcnow().isoformat().replace(":", "")
#         output_dir = self.config["sysconfig"].get("output_dir", None)
#         eval_id = f"{output_dir}_{date_time}" if output_dir else date_time

#         self.config["eval_id"] = eval_id
#         self.output_dir = os.path.join(self.host_paths.output_dir, eval_id) # Used in _cleanup()
#         self.tmp_dir = os.path.join(self.host_paths.tmp_dir, eval_id)       # Used in _cleanup()

#         # Retrieve environment variables that should be used in evaluation
#         log.info("Retrieving Environment Variables")
#         self.config.update({
#             "ARMORY_GITHUB_TOKEN": os.getenv("ARMORY_GITHUB_TOKEN", default=""),
#             "ARMORY_PRIVATE_S3_ID": os.getenv("ARMORY_PRIVATE_S3_ID", default=""),
#             "ARMORY_PRIVATE_S3_KEY": os.getenv("ARMORY_PRIVATE_S3_KEY", default=""),
#             "ARMORY_INCLUDE_SUBMISSION_BUCKETS": os.getenv(
#                 "ARMORY_INCLUDE_SUBMISSION_BUCKETS", default=""
#             ),
#             "VERIFY_SSL": self.armory_global_config["verify_ssl"] or False,
#             "NVIDIA_VISIBLE_DEVICES": self.config["sysconfig"].get("gpus", None),
#             "PYTHONHASHSEED": self.config["sysconfig"].get("set_pythonhashseed", "0"),
#             "TORCH_HOME": paths.HostPaths().pytorch_dir,
#             environment.ARMORY_VERSION: armory.__version__,
#             # "HOME": "/tmp",
#         })
#         self.config.update(os.environ.copy())
