import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    try:
        train_path, test_path = DataIngestion().initiate_data_ingestion()

        train_arr, test_arr, preprocessor_path, feature_names = (
            DataTransformation().initiate_data_transformation(train_path, test_path)
        )

        score = ModelTrainer().initiate_model_training(train_arr, test_arr)
        logging.info(f"Training pipeline completed successfully, final PR-AUC: {score:.4f}")
        print(f"Training complete. PR-AUC: {score:.4f}")

        X_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        evaluator = ModelEvaluation()
        y_proba = evaluator.evaluate("artifacts/model.pkl", X_test, y_test)

        evaluator.generate_shap_summary("artifacts/model.pkl", X_test, feature_names)

        best_threshold, best_profit = evaluator.cost_threshold_analysis(y_test, y_proba)
        print(f"Optimal approval threshold: {best_threshold}, expected profit: {best_profit}")

    except Exception as e:
        raise CustomException(e, sys)