from flask import Flask, render_template, request, jsonify
from datetime import datetime
from src.inference_service import InferenceService
from src.logger import get_logger
from src.custom_exception import CustomException

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

logger = get_logger(__name__)

# Initialize inference service
try:
    inference_service = InferenceService()
    logger.info("Flask app initialized with inference service")
except Exception as e:
    logger.error(f"Failed to initialize inference service: {e}")
    inference_service = None


@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint for Kubernetes"""
    if inference_service is None:
        return jsonify({
            "status": "unhealthy",
            "message": "Model not loaded"
        }), 503
    
    return jsonify({
        "status": "healthy",
        "message": "Service is running",
        "model_type": type(inference_service.model).__name__
    }), 200


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Get model metadata (available operation modes, etc.)"""
    try:
        if inference_service is None:
            return jsonify({"error": "Service not initialized"}), 503
        
        metadata = inference_service.get_metadata()
        return jsonify(metadata), 200
    
    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if service is initialized
        if inference_service is None:
            return jsonify({
                "success": False,
                "error": "Service not initialized",
                "message": "Model failed to load"
            }), 503
        
        # Get input data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Add timestamp if not provided
        if "Timestamp" not in data or not data["Timestamp"]:
            data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert numeric fields
        numeric_fields = [
            'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
            'Network_Latency_ms', 'Packet_Loss_%', 
            'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
            'Predictive_Maintenance_Score', 'Error_Rate_%'
        ]
        
        for field in numeric_fields:
            if field in data:
                try:
                    data[field] = float(data[field])
                except (ValueError, TypeError) as e:
                    return jsonify({
                        "success": False,
                        "error": f"Invalid value for {field}",
                        "message": f"{field} must be a number"
                    }), 400
        
        # Make prediction
        result = inference_service.predict(data)
        
        logger.info(f"Prediction request successful")
        
        return jsonify({
            "success": True,
            "prediction": result,
            "input_data": data
        }), 200
        
    except CustomException as e:
        logger.error(f"Custom exception in prediction: {e}")
        return jsonify({
            "success": False,
            "error": "Prediction error",
            "message": str(e)
        }), 400
    
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for external requests (same as /predict)"""
    return predict()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)