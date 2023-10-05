# app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["file"]

        # Read the uploaded file as a pandas DataFrame
        data = pd.read_csv(file)

        # Perform predictions using the trained model
        predictions = model.predict(data)

        # Convert the predictions to a list
        predictions = predictions.tolist()

        return render_template("result.html", predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
