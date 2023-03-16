"""convenient pre-fabricated "canned" armory evaluation experiments"""

from charmory.evaluation import (
    Attack,
    Dataset,
    Evaluation,
    MetaData,
    Metric,
    Model,
    Scenario,
    SysConfig,
)


def cifar10_baseline() -> Evaluation:
    return Evaluation(
        _metadata=MetaData(
            name="cifar_baseline",
            description="Baseline cifar10 image classification",
            author="msw@example.com",
        ),
        model=Model(
            function="armory.baseline_models.pytorch.cifar:get_art_model",
            model_kwargs={},
            wrapper_kwargs={},
            weights_file=None,
            fit=True,
            fit_kwargs={"nb_epochs": 20},
        ),
        scenario=Scenario(
            function="armory.scenarios.image_classification:ImageClassificationTask",
            kwargs={},
        ),
        dataset=Dataset(
            function="armory.data.datasets:cifar10",
            framework="numpy",
            batch_size=64
        ),
        attack=Attack(
            function="art.attacks.evasion:ProjectedGradientDescent",
            kwargs={
                "batch_size": 1,
                "eps": 0.031,
                "eps_step": 0.007,
                "max_iter": 20,
                "num_random_init": 1,
                "random_eps": False,
                "targeted": False,
                "verbose": False
            },
            knowledge="white",
            use_label=True,
            type=None,
        ),
        defense=None,
        metric=Metric(
            profiler_type="basic",
            supported_metrics=["accuracy"],
            perturbation=["linf"],
            task=["categorical_accuracy"],
            means=True,
            record_metric_per_sample=False,
        ),
        sysconfig=SysConfig(gpus=["all"], use_gpu=True),
    )
