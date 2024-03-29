from unittest.mock import MagicMock

from art.attacks import EvasionAttack
from art.estimators import BaseEstimator
import pytest
from torch.utils.data.dataloader import DataLoader

import charmory.evaluation


@pytest.fixture
def evaluation_model():
    return charmory.evaluation.Model(
        name="test",
        model=MagicMock(spec=BaseEstimator),
    )


@pytest.fixture
def data_loader():
    return MagicMock(spec=DataLoader)


@pytest.fixture
def evaluation_dataset(data_loader):
    return charmory.evaluation.Dataset(
        name="test",
        test_dataset=data_loader,
    )


@pytest.fixture
def evaluation_attack():
    attack = MagicMock(spec=EvasionAttack)
    attack.targeted = False
    return charmory.evaluation.Attack(name="test", attack=attack)


@pytest.fixture
def evaluation_metric():
    return charmory.evaluation.Metric()


@pytest.fixture
def evaluation_sysconfig():
    return charmory.evaluation.SysConfig(gpus=["all"], use_gpu=True)


@pytest.fixture
def evaluation(
    evaluation_model,
    evaluation_dataset,
    evaluation_attack,
    evaluation_metric,
    evaluation_sysconfig,
):
    return charmory.evaluation.Evaluation(
        name="test",
        description="test evaluation",
        author=None,
        model=evaluation_model,
        dataset=evaluation_dataset,
        attack=evaluation_attack,
        metric=evaluation_metric,
        sysconfig=evaluation_sysconfig,
    )
