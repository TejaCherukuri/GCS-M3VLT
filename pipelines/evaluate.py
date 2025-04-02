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
    test_data = data_loader.load_data(config.test_file)

    logging.info("Test Dataset loaded successfully.")

    # Create the MultiModalMedicalTransformer model
    gcsm3vlt_model = MedicalImageCaptioningModel(config=config, vectorization=vectorization)

    # Load model weights
    gcsm3vlt_model.load_weights(config.checkpoint_filepath)

    logging.info("Trained model loaded successfully.")

    # Evaluate model
    gcsm3vlt_model.evaluate_model(test_data, limit=5)

    logging.info("Model evaluated successfully.")

except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise CustomException(e, sys)