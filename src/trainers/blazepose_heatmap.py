import os
import pathlib

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from ..model_phase import ModelPhase
from ..models.keypoint_detection.blazepose import BlazePose
from ..data.mpii_datagen import MPIIDataGen


def train(config):
    """Train model

    Args:
        config (dict): Training configuration from configuration file
    """

    train_config = config["train"]
    model_config = config["model"]

    # Initialize model
    model = BlazePose(
        model_config["num_joints"], ModelPhase(model_config["model_phase"])).build_model()
    model.compile(optimizer=tf.optimizers.Adam(train_config["learning_rate"]),
                  loss="binary_crossentropy")

    # Load pretrained model
    if train_config["load_weights"]:
        print("Loading model weights: " +
              train_config["pretrained_weights_path"])
        model.load_weights(train_config["pretrained_weights_path"])

    # Create experiment folder
    exp_path = os.path.join("experiments/{}".format(config["experiment_name"]))
    pathlib.Path(exp_path).mkdir(parents=True, exist_ok=True)

    # Define the callbacks
    tb_log_path = os.path.join(exp_path, "tb_logs")
    tb = TensorBoard(log_dir=tb_log_path, write_graph=True)
    model_folder_path = os.path.join(exp_path, "models")
    pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)
    mc = ModelCheckpoint(filepath=os.path.join(
        model_folder_path, "model_ep{epoch:03d}.h5"), save_weights_only=True, save_format="h5", verbose=1)

    # Load data
    train_dataset = MPIIDataGen(
        config["data"]["train_images"],
        config["data"]["train_labels"],
        (model_config["im_height"], model_config["im_width"]),
        (128, 128),
        is_train=True)
    train_datagen = train_dataset.generator(train_config["train_batch_size"], 1, sigma=2, with_meta=False, is_shuffle=True,
                  rot_flag=True, scale_flag=True, flip_flag=True)

    val_dataset = MPIIDataGen(
        config["data"]["val_images"],
        config["data"]["val_labels"],
        (model_config["im_height"], model_config["im_width"]),
        (128, 128),
        is_train=False)
    val_datagen = val_dataset.generator(train_config["val_batch_size"], 1, sigma=2, with_meta=False, is_shuffle=False,
                  rot_flag=False, scale_flag=False, flip_flag=False)

    # Train
    model.fit(train_datagen,
              epochs=train_config["nb_epochs"],
              steps_per_epoch=train_dataset.get_dataset_size() // train_config["train_batch_size"],
              validation_data=val_datagen,
              validation_steps=val_dataset.get_dataset_size() // train_config["val_batch_size"],
              callbacks=[tb, mc],
              verbose=1
              )


def load_model(config, model_path):
    """Load pretrained model

    Args:
        config (dict): Model configuration
        model (str): Path to h5 model to be tested
    """

    model_config = config["model"]

    # Initialize model and load weights
    model = BlazePose(
        model_config["num_joints"], ModelPhase(model_config["model_phase"])).build_model()
    model.compile()
    model.load_weights(model_path)

    return model