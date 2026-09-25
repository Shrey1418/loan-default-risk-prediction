import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = FastAPI()

class ApplicationData(BaseModel):
    LIMIT_BAL: float
    AGE: int
    EDUCATION: int
    MARRIAGE: int
    SEX: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.post("/predict")
def predict(data: ApplicationData):
    try:
        df = pd.DataFrame([data.dict()])
        prob = PredictPipeline().predict(df)[0]
        decision = "REJECT" if prob > 0.30 else "APPROVE"
        logging.info(f"API Prediction: prob={prob:.4f}, decison={decision}")
        return {"default_probability": float(prob), "decision": decision}
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail="Prediction failed - check server logs")