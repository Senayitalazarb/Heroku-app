import os
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the input data from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])

        # Scale the input features
        input_features = scaler.transform([[feature1, feature2, feature3, feature4]])

        # Make the prediction
        prediction = model.predict(input_features)

        return render_template('index.html', prediction=prediction[0])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
