import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, processed_data_path, model_output_path):
        self.processed_path = processed_data_path
        self.model_path = model_output_path
        self.clf = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
        os.makedirs(self.model_path, exist_ok=True)
        logger.info("Model Training initialized...")
    
    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_path, "y_test.pkl"))
            
            logger.info(f"Training data loaded: {self.X_train.shape}")
            logger.info(f"Test data loaded: {self.X_test.shape}")
        
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", e)
        
    def train_model(self):
        try:
            self.clf = LogisticRegression(random_state=42, max_iter=1000)
            self.clf.fit(self.X_train, self.y_train)
            
            # Save model
            joblib.dump(self.clf, os.path.join(self.model_path, "model.pkl"))
            logger.info("Model trained and saved successfully")
        
        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException("Failed to train model", e)
        
    def evaluate_model(self):
        try:
            # Predictions
            y_pred = self.clf.predict(self.X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")
            
            logger.info("=" * 50)
            logger.info("MODEL EVALUATION RESULTS")
            logger.info("=" * 50)
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info("=" * 50)
            
            # Detailed classification report
            report = classification_report(self.y_test, y_pred)
            logger.info(f"\nClassification Report:\n{report}")
            
        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)
        
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()
        logger.info("Model training pipeline completed successfully")


if __name__ == "__main__":
    trainer = ModelTraining("artifacts/processed/", "artifacts/models/")
    trainer.run()