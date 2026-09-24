import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def predict(self, features: pd.DataFrame):
        try:
            preprocessor = load_object("artifacts/preprocessor.pkl")
            model = load_object("artifacts/model.pkl")
            data_scaled = preprocessor.transform(features)
            prob = model.predict_proba(data_scaled)[:, 1]
            logging.info(f"Prediction generated for {len(features)} record(s)")
            return prob
        except Exception as e:
            raise CustomException(e, sys)