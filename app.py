
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the metadata dictionary containing both model and feature list
with open('bike_model_metadata.pkl', 'rb') as f:
    payload = pickle.load(f)
    model = payload['model']
    features_list = payload['features']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/info', methods=['GET'])
def info():
    # Returns the list of feature names the model expects
    return jsonify({'features': features_list})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Map incoming JSON data to the exact ordered feature set
        input_features = []
        for feat in features_list:
            input_features.append(float(data['features'][feat]))
        
        prediction = model.predict(np.array(input_features).reshape(1, -1))
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
