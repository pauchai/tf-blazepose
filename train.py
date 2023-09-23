import os
import pathlib
import shutil
import argparse
import importlib
import json
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--conf_file', default="config.json",
    help='Configuration file')
parser.add_argument(
    '-e',
    '--experiments_dir', default="experiments",
    help='Experiments dir')

args = parser.parse_args()

# Open and load the config json
with open(args.conf_file) as config_buffer:
    config = json.loads(config_buffer.read())

# Create experiment folder and copy configuration file
exp_folder = os.path.join(args.experiments_dir, config["experiment_name"])
pathlib.Path(exp_folder).mkdir(parents=True, exist_ok=True)
shutil.copy(args.conf_file, exp_folder)

# Train model
trainer = importlib.import_module("src.trainers.{}".format(config["trainer"]))
trainer.train(config, exp_folder)
