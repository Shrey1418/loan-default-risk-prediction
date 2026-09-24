import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_path = "artifacts/preprocessor.pkl"

    def clean_data(self, df):
        try:
            df["EDUCATION"] = df["EDUCATION"].replace([0,5,6],"other")
            df["MARRIAGE"] = df["MARRIAGE"].replace(0,"other")
            logging.info("Cleaned invalid EDUCATION and MARRIAGE codes")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def engineer_features(self, df):
        try:
            bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
            pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
            df["Bill_to_Limit_Ratio"] = df[bill_cols].mean(axis=1) / df["LIMIT_BAL"]
            df["Payment_Ratio"] = df[pay_cols].sum(axis=1) / (df[bill_cols].sum(axis=1) + 1)
            df["Has_Delay_History"] = (df["Total_Delay_Score"] > 0).astype(int)
            logging.info("Feature engineering complete")
            return df
        except Exception as e:
            raise CustomException(e,sys)

    def get_preprocessor_object(self, numeric_features, categorical_features):
        try:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numeric_features),
                (("cat_pipeline", cat_pipeline, categorical_features))
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_path)
            train_df = self.engineer_features(train_df)
            test_df = self.engineer_features(test_path)

            target_column = "default_payment_next_month"
            numeric_features = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                                "Bill_to_Limit_Ratio", "Payment_Ratio", "Has_Delay_History", "Total_Delay_Score"]
            categorical_features = ["EDUCATION", "MARRIAGE", "SEX", "Credit_Limit_Tier"]

            preprocessor = self.get_preprocessor_object(numeric_features, categorical_features)

            X_train = train_df[numeric_features + categorical_features]
            y_train = train_df[target_column]
            X_test = test_df[numeric_features + categorical_features]
            y_test = test_df[target_column]

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Preprocessing pipeline fit and applied")

            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
            logging.info(f"SMOTE applied — before: {X_train_transformed.shape}, after: {X_train_resampled.shape}")

            save_object(self.preprocessor_obj_path, preprocessor)

            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            return train_arr, test_arr, self.preprocessor_obj_path

        except Exception as e:
            logging.error("Data transformation failed")
            raise CustomException(e,sys) 