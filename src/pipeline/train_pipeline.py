import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    try:
        train_path, test_path = DataIngestion().initiate_data_ingestion()
        train_arr, test_arr, preprocessor_path = DataTransformation().initiate_data_transformation(train_path, test_path)
        score = ModelTrainer().initiate_model_training(train_arr, test_arr)
        logging.info(f"Triang pipeline completed sucessfuly, final PR-AUC: {score:.4f}")
        print(f"Training complete. PR-AUC: {score:.4f}")
    except Exception as e:
        raise CustomException(e, sys)