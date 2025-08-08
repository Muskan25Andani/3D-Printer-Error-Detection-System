from flask import Flask, request, render_template
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the saved model, label encoder, and scaler
try:
    logger.info("Loading model, label encoder, and scaler...")
    best_model = joblib.load('best_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    logger.info("Model, label encoder, and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or components: {str(e)}")
    raise Exception(f"Failed to load model or components: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    input_values = None
    
    if request.method == 'POST':
        try:
            # Get input values from form
            x_input = float(request.form['x_input'])
            y_input = float(request.form['y_input'])
            z_input = float(request.form['z_input'])
            
            # Store input values for display
            input_values = [x_input, y_input, z_input]
            logger.info(f"Received input: X={x_input}, Y={y_input}, Z={z_input}")
            
            # Prepare input for prediction
            custom_input = np.array([[x_input, y_input, z_input]])
            
            # Log expected feature count
            expected_features = getattr(scaler, 'n_features_in_', 3)  # Default to 3 if not available
            logger.info(f"Expected number of features: {expected_features}")
            
            # Pad zeros if model was trained on more features
            if custom_input.shape[1] != expected_features:
                logger.info(f"Padding input from {custom_input.shape[1]} to {expected_features} features")
                zeros = np.zeros((1, expected_features - 3))
                custom_input = np.hstack((custom_input, zeros))
            
            # Scale the input
            logger.info("Scaling input data...")
            custom_input_scaled = scaler.transform(custom_input)
            
            # Make prediction
            logger.info("Making prediction...")
            custom_pred = best_model.predict(custom_input_scaled)
            prediction = label_encoder.inverse_transform(custom_pred)[0].upper()
            logger.info(f"Prediction: {prediction}")
            
        except ValueError as ve:
            logger.error(f"ValueError during prediction: {str(ve)}")
            prediction = f"Error: Invalid input values - {str(ve)}"
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {str(e)}")
            prediction = f"Error: Prediction failed - {str(e)}"
    
    return render_template('index.html', prediction=prediction, input_values=input_values)

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True)