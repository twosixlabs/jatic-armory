"""
Based on `track.py`
"""
import os
import sys

import charmory.canned
from charmory.engine import Engine


def configure_environment():
    """
    Setup a general machine learning development environment.
    """
    print("Delayed imports and dependency configuration.")

    try:
        print("Importing and configuring torch, tensorflow, and art, if available. ")
        print("This may take some time.")

        # import torch before tensorflow to ensure torch.utils.data.DataLoader can utilize
        # all CPU resources when num_workers > 1
        import art
        import tensorflow as tf
        import torch  # noqa: F401

        from armory.paths import HostPaths

        # Handle ART configuration by setting the art data
        # path if art can be imported in the current environment
        art.config.set_data_path(os.path.join(HostPaths().saved_model_dir, "art"))

        if gpus := tf.config.list_physical_devices("GPU"):
            # Currently, memory growth needs to be the same across GPUs
            # From: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(
                "Setting tf.config.experimental.set_memory_growth to True on all GPUs"
            )

    except RuntimeError:
        print("Import armory before initializing GPU tensors")
        raise
    except ImportError:
        pass


def show_mlflow_experiement(experiment_id):
    experiment = mlflow.get_experiment(experiment_id)
    print(f"Experiment: {experiment.name}")
    print(f"tags: {experiment.tags}")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
    print(f"Creation Time: {experiment.creation_time}")

    def run(self):
        """fake an evaluation to demonstrate mlflow tracking."""
        metadata = self.evaluation._metadata
        log.info("Starting mlflow run:")
        show_mlflow_experiement(self.experiment_id)
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            description=metadata.description,
            tags={
                "author": self.evaluation._metadata.author,
            },
        )

        # fake variable epsilon and results
        import random

        epsilon = random.random()
        result = {"benign": epsilon, "adversarial": 1 - epsilon}
        self.evaluation.attack.kwargs["eps"] = epsilon

        for key, value in self.evaluation.flatten():
            if key.startswith("_metadata."):
                continue
            mlflow.log_param(key, value)

        for k, v in result.items():
            mlflow.log_metric(k, v)

        mlflow.end_run()
        return result


def main():
    print("Armory: Example Programmatic Entrypoint for Scenario Execution")
    # configure_environment()

    print("Starting demo")
    mnist = charmory.canned.mnist_baseline()
    evaluator = Engine(mnist)
    evaluator.run()

    print("mnist experiment results tracked")


if __name__ == "__main__":
    sys.exit(main())
