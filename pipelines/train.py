import sys
from utils.config import Config, LRSchedule, GCRMSprop
from components import DataLoader
from components import MedicalImageCaptioningModel
from utils import logging, CustomException

try:
    # Initialize the Config object
    config = Config()

    # Create Datasets
    data_loader = DataLoader(config)
    train_dataset, valid_dataset, test_dataset, vectorization = data_loader.load_datasets()

    logging.info("Dataset loaded successfully.")

    # Create the MultiModalMedicalTransformer model
    gcsm3vlt_model = MedicalImageCaptioningModel(config=config, vectorization=vectorization)

    # Create a learning rate schedule and optimizer
    num_train_steps = len(train_dataset) * config.epochs
    num_warmup_steps = num_train_steps // config.batch_size

    lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)
    optimizer = GCRMSprop(lr_schedule)

    # Compile the model
    gcsm3vlt_model.compile(optimizer=optimizer, loss=config.cross_entropy)

    logging.info("Model created successfully.")

    # Fit the model
    gcsm3vlt_model.fit(train_dataset, epochs=config.epochs, validation_data=test_dataset, verbose=1)

    logging.info("Model trained successfully.")

    # Save model weights
    gcsm3vlt_model.save_weights(config.checkpoint_filepath, save_format="tf")

    logging.info("Model saved successfully.")

except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise CustomException(e, sys)