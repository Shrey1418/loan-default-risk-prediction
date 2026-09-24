import os
import sys
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self):
        self.raw_data_path = "artifacts/raw.csv"
        self.db_path = "artifacts/loan_data.db"
        self.train_data_path = "artifacts/train.csv"
        self.test_data_path = "artifacts/test.csv"

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")
        try:
            df = pd.read_excel("notebooks/data/default_of_credit_card_clients.xls", header=1)
            df.rename(columns={"default payment next month": "default_payment_next_month"}, inplace=True)
            logging.info(f"Read raw data with shape {df.shape}")

            os.makedirs("artifacts", exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)

            conn = sqlite3.connect(self.db_path)
            df.to_sql("applicants", conn, if_exists="replace", index=False)
            logging.info("Loaded raw data into SQLite")

            query = """
            SELECT *,
                (PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6) AS Total_Delay_Score,
                CASE
                    WHEN LIMIT_BAL < 50000 THEN 'low'
                    WHEN LIMIT_BAL < 150000 THEN 'medium'
                    WHEN LIMIT_BAL < 300000 THEN 'high'
                    ELSE 'very_high'
                END AS Credit_Limit_Tier
            FROM applicants
            """
            df_engineered = pd.read_sql(query, conn)
            conn.close()
            logging.info("SQL feature engineering complete")

            train_set, test_set = train_test_split(
                df_engineered, test_size=0.2, random_state=42,
                stratify=df_engineered["default_payment_next_month"]
            )
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)
            logging.info("Train/test split saved")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logging.error("Data ingestion failed")
            raise CustomException(e,sys)
