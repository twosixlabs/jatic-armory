# Generated by CodiumAI

import pytest

from charmory.evaluation import Dataset

"""
Code Analysis:
- The class 'Dataset' is a dataclass that represents a dataset configuration for an Armory experiment.
- It has three fields: 'function', 'framework', and 'batch_size'.
- 'function' is a reference to a python method that will be used to load the dataset.
- 'framework' is a literal that specifies the framework used to load the dataset, which can be either 'tf', 'torch', or 'numpy'.
- 'batch_size' is an integer that specifies the batch size to use when loading the dataset.
- The class is used to configure the dataset for an Armory experiment, allowing the user to specify the function, framework, and batch size to use when loading the dataset.
- The 'function' field is a reference to a python method that will be used to load the dataset. This allows the user to specify a custom function for loading the dataset, rather than using the default function provided by Armory.
- The 'framework' field specifies the framework used to load the dataset. This is important because different frameworks have different APIs for loading datasets, and Armory needs to know which API to use.
- The 'batch_size' field specifies the batch size to use when loading the dataset. This is important because it affects the memory usage and training speed of the model.
- Overall, the 'Dataset' class provides a convenient way for users to configure the dataset for an Armory experiment, allowing them to specify the function, framework, and batch size to use when loading the dataset.
"""


class TestDataset:
    # Tests that a dataset object can be created with valid function, framework, and batch_size inputs. tags: [happy path]
    def test_create_dataset_valid_inputs(self):
        dataset = Dataset(
            function="armory.data.datasets.cifar10", framework="tf", batch_size=32
        )
        assert dataset.function == "armory.data.datasets.cifar10"
        assert dataset.framework == "tf"
        assert dataset.batch_size == 32

    # Tests that the function field of a dataset object can be accessed. tags: [happy path]
    def test_access_function_field(self):
        dataset = Dataset(
            function="armory.data.datasets.cifar10", framework="tf", batch_size=32
        )
        assert dataset.function == "armory.data.datasets.cifar10"

    # Tests that a dataset object cannot be created with an invalid function input. tags: [edge case]
    def test_create_dataset_invalid_function_input(self):
        with pytest.raises(TypeError):
            Dataset(function=123)  # type: ignore

    # Tests that the framework field of a dataset object can be accessed. tags: [happy path]
    def test_access_framework_field(self):
        dataset = Dataset(
            function="armory.data.datasets.cifar10", framework="tf", batch_size=32
        )
        assert dataset.framework == "tf"

    # Tests that the batch_size field of a dataset object can be accessed. tags: [happy path]
    def test_access_batch_size_field(self):
        dataset = Dataset(
            function="armory.data.datasets.cifar10", framework="tf", batch_size=32
        )
        assert dataset.batch_size == 32
