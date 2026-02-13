import joblib
import pandas as pd
from datetime import datetime
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class InferenceService:
    def __init__(self, model_path="artifacts/models", processed_path="artifacts/processed"):
        try:
            # Load model
            self.model = joblib.load(f"{model_path}/model.pkl")
            
            # Load preprocessing artifacts
            self.scaler = joblib.load(f"{processed_path}/scaler.pkl")
            self.label_encoders = joblib.load(f"{processed_path}/label_encoders.pkl")
            self.features = joblib.load(f"{processed_path}/features.pkl")
            
            logger.info("Inference service initialized successfully")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Features: {self.features}")
            logger.info(f"Operation_Mode classes: {self.label_encoders['Operation_Mode'].classes_}")
            logger.info(f"Efficiency_Status classes: {self.label_encoders['Efficiency_Status'].classes_}")
        
        except Exception as e:
            logger.error(f"Error initializing inference service: {e}")
            raise CustomException("Failed to initialize inference service", e)
    
    def predict(self, input_data):
        """
        Make prediction on input data
        
        Args:
            input_data (dict): Dictionary containing input features
            
        Returns:
            dict: Prediction results with efficiency status and probabilities
        """
        try:
            logger.info(f"Received prediction request: {input_data}")
            
            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            
            # Feature engineering (same as data_processing.py)
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
                df["Year"] = df["Timestamp"].dt.year
                df["Month"] = df["Timestamp"].dt.month
                df["Day"] = df["Timestamp"].dt.day
                df["Hour"] = df["Timestamp"].dt.hour
                df.drop(columns=["Timestamp"], inplace=True)
            else:
                # Use current timestamp if not provided
                now = datetime.now()
                df["Year"] = now.year
                df["Month"] = now.month
                df["Day"] = now.day
                df["Hour"] = now.hour
            
            # Drop Machine_ID if present
            if "Machine_ID" in df.columns:
                df.drop(columns=["Machine_ID"], inplace=True)
            
            # Encode Operation_Mode
            if "Operation_Mode" in df.columns:
                df["Operation_Mode"] = self.label_encoders["Operation_Mode"].transform(
                    df["Operation_Mode"]
                )
            
            # Select features in correct order
            X = df[self.features]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            
            # Decode prediction
            efficiency_status = self.label_encoders["Efficiency_Status"].inverse_transform(
                [prediction]
            )[0]
            
            # Prepare result
            result = {
                "efficiency_status": efficiency_status,
                "confidence": float(max(prediction_proba)),
                "probabilities": {
                    str(cls): float(prob) 
                    for cls, prob in zip(
                        self.label_encoders["Efficiency_Status"].classes_, 
                        prediction_proba
                    )
                }
            }
            
            logger.info(f"Prediction successful: {efficiency_status} (confidence: {result['confidence']:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise CustomException("Prediction failed", e)
    
    def get_metadata(self):
        """Return model metadata"""
        return {
            "operation_modes": self.label_encoders["Operation_Mode"].classes_.tolist(),
            "efficiency_statuses": self.label_encoders["Efficiency_Status"].classes_.tolist(),
            "features": self.features
        }