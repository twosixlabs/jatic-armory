import argparse
from pprint import pprint

import albumentations as A
import art.attacks.evasion
from art.estimators.object_detection import PyTorchObjectDetector
import jatic_toolbox
import numpy as np
import torch
from torchvision.ops import box_convert
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from armory.art_experimental.attacks.patch import AttackWrapper
from armory.metrics.compute import BasicProfiler
from charmory.data import ArmoryDataLoader, JaticObjectDetectionDataset
from charmory.engine import LightningEngine
from charmory.evaluation import Attack, Dataset, Evaluation, Metric, Model, SysConfig
from charmory.model.object_detection import YolosTransformer
from charmory.tasks.object_detection import ObjectDetectionTask
from charmory.track import track_init_params, track_params


def get_cli_args():
    parser = argparse.ArgumentParser(
        description="Run COCO object detection example using models and datasets from the JATIC toolbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--export-every-n-batches",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--num-batches",
        default=20,
        type=int,
    )
    return parser.parse_args()


@track_params(prefix="main")
def main(batch_size, export_every_n_batches, num_batches):
    ###
    # Model
    ###
    model = track_params(AutoModelForObjectDetection.from_pretrained)(
        pretrained_model_name_or_path="hustvl/yolos-tiny"
    )
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    transformer = YolosTransformer(model, image_processor)

    detector = track_init_params(PyTorchObjectDetector)(
        transformer,
        channels_first=True,
        input_shape=(3, 512, 512),
        clip_values=(0.0, 1.0),
        attack_losses=(
            "cardinality_error",
            "loss_bbox",
            "loss_ce",
            "loss_giou",
        ),
    )

    ###
    # Dataset
    ###
    dataset = track_params(jatic_toolbox.load_dataset)(
        provider="huggingface",
        dataset_name="rafaelpadilla/coco2017",
        task="object-detection",
        split="val",
        category_key="label",
    )

    # Have to filter out non-RGB images
    def filter(sample):
        shape = np.asarray(sample["image"]).shape
        return len(shape) == 3 and shape[2] == 3

    print(f"Dataset length prior to filtering: {len(dataset)}")
    dataset._dataset = dataset._dataset.filter(filter)
    print(f"Dataset length after filtering: {len(dataset)}")

    # Resize and pad images to 512x512
    img_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(
                min_height=512,
                min_width=512,
                border_mode=0,
                value=(0, 0, 0),
            ),
            A.ToFloat(max_value=255),  # Scale to [0,1]
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
        ),
    )

    def transform(sample):
        transformed = dict(image=[], objects=[])
        for i in range(len(sample["image"])):
            transformed_img = img_transforms(
                image=np.asarray(sample["image"][i]),
                bboxes=sample["objects"][i]["bbox"],
                labels=sample["objects"][i]["label"],
            )
            # Transpose from HWC to CHW
            transformed["image"].append(transformed_img["image"].transpose(2, 0, 1))
            transformed["objects"].append(
                dict(
                    bbox=transformed_img["bboxes"],
                    label=transformed_img["labels"],
                )
            )
        for obj in transformed["objects"]:
            if len(obj.get("bbox", [])) > 0:
                obj["bbox"] = box_convert(
                    torch.tensor(obj["bbox"]), "xywh", "xyxy"
                ).numpy()
        return transformed

    dataset.set_transform(transform)

    dataloader = ArmoryDataLoader(
        JaticObjectDetectionDataset(dataset), batch_size=batch_size
    )

    ###
    # Evaluation
    ###
    eval_dataset = Dataset(
        name="coco",
        test_dataset=dataloader,
    )

    eval_model = Model(
        name="faster-rcnn-resnet50",
        model=detector,
    )

    patch = track_init_params(art.attacks.evasion.RobustDPatch)(
        detector,
        patch_shape=(3, 50, 50),
        patch_location=(231, 231),  # middle of 512x512
        batch_size=1,
        sample_size=10,
        learning_rate=0.01,
        max_iter=20,
        targeted=False,
        verbose=False,
    )

    eval_attack = Attack(
        name="RobustDPatch",
        attack=AttackWrapper(patch),
        use_label_for_untargeted=False,
    )

    eval_metric = Metric(
        profiler=BasicProfiler(),
    )

    eval_sysconfig = SysConfig(
        gpus=["all"],
        use_gpu=True,
    )

    evaluation = Evaluation(
        name="coco-yolos-object-detection",
        description="COCO object detection using YOLO from HuggingFace",
        author="",
        dataset=eval_dataset,
        model=eval_model,
        attack=eval_attack,
        metric=eval_metric,
        sysconfig=eval_sysconfig,
    )

    ###
    # Engine
    ###

    task = ObjectDetectionTask(
        evaluation,
        export_every_n_batches=export_every_n_batches,
        class_metrics=False,
    )
    engine = LightningEngine(task, limit_test_batches=num_batches)
    results = engine.run()

    pprint(results)


if __name__ == "__main__":
    main(**vars(get_cli_args()))
