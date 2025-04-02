import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from typing import Dict, Any
from keras.optimizers import RMSprop

class Config:
    def __init__(self):
        # Data paths
        self.data_dir = "/kaggle/input/deepeyenet"
        self.train_file = "/kaggle/input/deepeyenet/DeepEyeNet_train.json"
        self.val_file = "/kaggle/input/deepeyenet/DeepEyeNet_valid.json"
        self.test_file = "/kaggle/input/deepeyenet/DeepEyeNet_test.json"
        self.result_file = "result_data.csv"

        # Image and text processing
        self.image_size = (356, 356)
        self.min_seq_length = 5
        self.max_seq_length = 50
        self.vocab_size = 5000
        self.strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~".replace("<", "").replace(">", "")
        self.num_captions_per_image = 1
        self.corpus_file = "./corpus.txt"

        # Model hyperparameters
        self.embed_dim = 1024
        self.ff_dim = 1024
        self.num_heads = 2

        # Training parameters
        self.epochs = 25
        self.batch_size = 32
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")
        self.checkpoint_filepath = "weights/weights.best.tf"

        # Reproducibility
        self.seed = 812
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

        # GradCAM++ settings
        self.gradcampp_layer = "attention_gate"
        self.gradcampp_dir = os.path.join("./", "gradcampp")

    def get_config(self):
        """
        Returns a JSON-serializable dictionary of the configuration.
        """
        return {
            "data_dir": self.data_dir,
            "train_file": self.train_file,
            "val_file": self.val_file,
            "test_file": self.test_file,
            "result_file": self.result_file,
            "image_size": self.image_size,
            "min_seq_length": self.min_seq_length,
            "max_seq_length": self.max_seq_length,
            "vocab_size": self.vocab_size,
            "strip_chars": self.strip_chars,
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "num_heads": self.num_heads,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "num_captions_per_image": self.num_captions_per_image,
            "checkpoint_filepath": self.checkpoint_filepath,
        }

    @classmethod
    def from_config(cls, config_dict):
        """
        Creates a Config object from a dictionary.
        """
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config

    def save(self, filepath):
        """
        Saves the configuration to a JSON file.
        """
        config_dict = self.get_config()
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, filepath):
        """
        Loads the configuration from a JSON file.
        """
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_config(config_dict)

# Learning Rate Scheduler for the optimizer
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )

    def get_config(self):
        return {
            "post_warmup_learning_rate": self.post_warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }


class GCRMSprop(RMSprop):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= ops.mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads