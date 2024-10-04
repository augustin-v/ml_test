from flask import Flask, request, jsonify
import joblib
import xgboost as xgb
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

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
    app.run(debug=True)
