"""
Based on `track.py`
"""
import datetime
import json
import os
import shutil
import sys

import armory
from armory import environment, paths
from armory.configuration import load_global_config
from armory.logs import log
from armory.utils.printing import bold, red

# from charmory.examples.mnist_evaluation import mnist_baseline
from charmory.examples.cifar10_evaluation import cifar10_baseline


def main():
    print("Armory: Example Programmatic Entrypoint for Scenario Execution")
    # demo_evaluation = mnist_baseline()
    demo_evaluation = cifar10_baseline()

    log.info(bold(f"Starting Demo for {red(demo_evaluation._metadata.name)}"))

    result = demo_evaluation.run()
    result["benign"] = id(demo_evaluation)

    if self.evaluation.attack:
        result["attack"] = id(demo_evaluation)

    log.info(bold("mnist experiment results tracked"))

    print(("=" * 64).center(128))

    print(__import__("json").dumps(demo_evaluation.asdict(), indent=4, sort_keys=True))
    print(("-" * 64).center(128))

    print(result)
    print(("=" * 64).center(128))

    return result


if __name__ == "__main__":
    sys.exit(main())


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
