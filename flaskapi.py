from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('RandomForest.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse request data
    data = request.get_json(force=True)
    
    # Ensure that all the required features are present in the incoming JSON
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    features = [data.get(feature) for feature in feature_names]

    if not all(features):
        return jsonify({'error': 'Missing features'}), 400

    prediction = model.predict([features])

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True)
