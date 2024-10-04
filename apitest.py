from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Enable CORS for your Flask app
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Load the trained model
model = xgb.XGBRegressor()
model.load_model('stark_chan_v2.json')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the POST request
    input_data = request.get_json()

    # Assuming the input is a list of feature values
    features = np.array(input_data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    # Run the Flask app on 0.0.0.0 and the default port
    app.run(host='0.0.0.0', port=5000, debug=True)  # Update host and port for deployment
