import os

from art.classifiers import KerasClassifier
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from armory import paths
from armory.data.utils import download_file_from_s3

num_classes = 45


def mean_std():
    resisc_mean = np.array(
        [0.36386173189316956, 0.38118692953271804, 0.33867067558870334,]
    )

    resisc_std = np.array([0.20350874, 0.18531173, 0.18472934])

    return resisc_mean, resisc_std


def preprocess_input_densenet121_resisc(img):
    # Model was trained with Caffe preprocessing on the images
    # load the mean and std of the [0,1] normalized dataset
    # Normalize images: divide by 255 for [0,1] range
    mean, std = mean_std()
    img_norm = img / 255.0
    # Standardize the dataset on a per-channel basis
    output_img = (img_norm - mean) / std
    return output_img


def preprocessing_fn(x: np.ndarray) -> np.ndarray:
    shape = (224, 224)  # Expected input shape of model
    output = []
    for i in range(x.shape[0]):
        im_raw = image.array_to_img(x[i])
        im = image.img_to_array(im_raw.resize(shape))
        output.append(im)
    output = preprocess_input_densenet121_resisc(np.array(output))
    return output


def make_densenet121_resisc_model(**model_kwargs) -> tf.keras.Model:
    # Load ImageNet pre-trained DenseNet
    model_notop = DenseNet121(
        include_top=False, weights=None, input_shape=(224, 224, 3)
    )

    # Add new layers
    x = GlobalAveragePooling2D()(model_notop.output)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Create graph of new model and freeze pre-trained layers
    new_model = Model(inputs=model_notop.input, outputs=predictions)

    for layer in new_model.layers[:-1]:
        layer.trainable = False
        if "bn" == layer.name[-2:]:  # allow batchnorm layers to be trainable
            layer.trainable = True

    # compile the model
    new_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return new_model


def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    model = make_densenet121_resisc_model(**model_kwargs)
    if weights_file:
        saved_model_dir = paths.docker().saved_model_dir
        filepath = os.path.join(saved_model_dir, weights_file)

        if not os.path.isfile(filepath):
            download_file_from_s3(
                "armory-public-data",
                f"model-weights/{weights_file}",
                f"{saved_model_dir}/{weights_file}",
            )

        model.load_weights(filepath)

    mean, std = mean_std()
    wrapped_model = KerasClassifier(
        model, clip_values=((0.0 - mean) / std, (1.0 - mean) / std), **wrapper_kwargs
    )
    return wrapped_model