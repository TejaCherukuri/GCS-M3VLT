import os
import sys
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K

from utils.config import Config, LRSchedule, GCRMSprop
from components import DataLoader
from components import MedicalImageCaptioningModel, GradCAMPP
from utils import logging, CustomException

class Predict:           
    def __init__(self, config):
        self.config = config
        self.model = self._load_model()
        self.gradcampp = GradCAMPP(self.model, config.gradcampp_layer)

    def _load_model(self):
        try:
            K.clear_session()

            # Initialize the Config object
            config = Config()

            # Create Datasets
            data_loader = DataLoader(config)
            # Open and read the Corpus file
            with open(config.corpus_file, "r", encoding="utf-8") as file:
                corpus = file.readlines()  # Read lines into a list
            vectorization = data_loader.build_vectorization(corpus)
            logging.info("Vectorization successful.")

            lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=215)
            optimizer = GCRMSprop(lr_schedule)
            # Create the MultiModalMedicalTransformer model
            model = MedicalImageCaptioningModel(config=config, vectorization=vectorization)
            # Compile the model
            model.compile(optimizer=optimizer, loss=config.cross_entropy)

            logging.info("Model created successfully.")

            # Create a checkpoint object
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

            # Restore the checkpoint
            checkpoint_path = config.checkpoint_filepath  # Ensure this is correctly set
            status = checkpoint.restore(checkpoint_path)
            status.expect_partial()  # Ignore any missing objects (useful if optimizer states are missing)

            logging.info("Model loaded successfully from checkpoint.")

            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise CustomException(e, sys)

    def load_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Unable to read image {img_path}.")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.config.image_size[0], self.config.image_size[1]), interpolation=cv2.INTER_CUBIC)
            logging.info(f"Image {img_path} loaded and preprocessed successfully.")
            return img

        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            raise CustomException(e, sys)

    def predict(self, img_path, keywords):
        try:
            img_name = img_path.split("/")[-1]
            img = self.load_image(img_path)
            caption = self.model.generate(img_path, keywords, img_verbose=False, beam_search=True)
            heatmap = self.gradcampp.heatmap(img, 0)
            self.gradcampp.show(img, heatmap, save_path = os.path.join(self.config.gradcampp_dir, img_name))
            return caption

        except Exception as e:
            logging.error(f"Error predicting image {img_path}: {e}")
            raise CustomException(e, sys)

## Inference Sample     
try:
    config = Config()
    pred = Predict(config)

    img_path = "/test_samples/group13-6.jpg"
    keywords = 'branch retinal artery occlusion (brao), pan-retinal photocoagulation (prp), hollenhorst plaque'
    caption = pred.predict(img_path, keywords)

    print(caption)

except Exception as e:
    logging.error(f"Error predicting image {img_path}: {e}")
    raise CustomException(e, sys)