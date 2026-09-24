import sys
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class ModelEvaluation:
    def evaluate(self, model_path, X_test, y_test):
        try:
            model = load_object(model_path)
            y_pred =  model.predict(X_test)
            y_proba =  model.predict_proba(X_test)[:, 1]

            logging.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
            logging.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            logging.info(f"PR-AUC: {average_precision_score(y_test, y_proba):.4f}")

            return y_proba
        except Exception as e:
            raise CustomException(e, sys)

    def generate_shap_summary(self, model_path, X_test, feature_names):
        try:
            model = load_object(model_path)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show = False)
            plt.savefig("artifacts/shap_summary.png", bbox_inches="tight")
            logging.info("SHAP summary plot saved")

        except Exception as e:
            raise CustomException (e, sys)

    def cost_threshold_analysis(self, y_test, y_proba, loss_per_default=4000, profit_per_approval=400):
        try:
            best_threshold, best_profit = 0.5, float("-inf")
            for threshold in [i / 100 for i in range(10, 91, 5)]:
                predictions = (y_proba >= threshold).astype(int)
                approvals = (predictions == 0).sum()
                missed_defaults = ((predictions == 0) & (y_test == 1)).sum()
                expected_profit = (approvals * profit_per_approval) - (missed_defaults * loss_per_default)
                if expected_profit > best_profit:
                    best_profit, best_threshold = expected_profit, threshold
            logging.info(f"Optimal threshold: {best_threshold}, expected profit: {best_profit}")
            return best_threshold, best_profit

        except Exception as e:
            raise CustomException(e, sys)