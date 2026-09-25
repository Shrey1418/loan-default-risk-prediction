import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def _engineer_features(self, df):
        # Same logic as data_ingestion.py's SQL step, done in pandas for a single request
        df["Total_Delay_Score"] = df["PAY_0"] + df["PAY_2"] + df["PAY_3"] + df["PAY_4"] + df["PAY_5"] + df["PAY_6"]

        def bucket_limit(limit_bal):
            if limit_bal < 50000:
                return "low"
            elif limit_bal < 150000:
                return "medium"
            elif limit_bal < 300000:
                return "high"
            else:
                return "very_high"
        df["Credit_Limit_Tier"] = df["LIMIT_BAL"].apply(bucket_limit)

        # Same logic as data_transformation.py's engineer_features
        bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
        pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
        df["Bill_to_Limit_Ratio"] = df[bill_cols].mean(axis=1) / df["LIMIT_BAL"]
        df["Payment_Ratio"] = df[pay_cols].sum(axis=1) / (df[bill_cols].sum(axis=1) + 1)
        df["Has_Delay_History"] = (df["Total_Delay_Score"] > 0).astype(int)

        return df

    def predict(self, features: pd.DataFrame):
        try:
            features = self._engineer_features(features)
            preprocessor = load_object("artifacts/preprocessor.pkl")
            model = load_object("artifacts/model.pkl")
            data_scaled = preprocessor.transform(features)
            prob = model.predict_proba(data_scaled)[:, 1]
            logging.info(f"Prediction generated for {len(features)} record(s)")
            return prob
        except Exception as e:
            raise CustomException(e, sys)