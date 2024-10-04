from flask import Flask, request, jsonify
import joblib
import xgboost as xgb
import numpy as np


app = Flask(__name__)

model = xgb.XGBRegressor()
model.load_model('stark_chan_v2.json')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    features = np.array(input_data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
