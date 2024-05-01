from flask import Flask, redirect, render_template, request
import pandas as pd
import joblib
import os
import numpy as np


app = Flask(__name__)


def classify(inputs):
    new_data = np.array(inputs).reshape(1, -1)

    model_folder = r"C:\Users\Lenovo\Desktop\project-1\models"
    model_file = "best_model.pkl"
    model_path = os.path.join(model_folder, model_file)
    model = joblib.load(model_path)

    prediction = model.predict(new_data)
    return prediction


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get values from the form
        age = float(request.form.get("Age"))
        gender = int(request.form.get('gender'))
        hours = int(request.form.get('hours'))
        working = int(request.form.get('working'))
        painful = int(request.form.get('painful'))
        gritty = int(request.form.get('gritty'))
        sensitive = int(request.form.get('sensitive'))
        TV = int(request.form.get('TV'))
        ac = int(request.form.get('ac'))
        humidity = int(request.form.get('humidity'))
        windy = int(request.form.get('windy'))
        driving = int(request.form.get('driving'))
        blurred = int(request.form.get('blurred'))
        poor = int(request.form.get('poor'))
        read = int(request.form.get('reading'))

        p = classify([age, gender, hours, sensitive, gritty, painful, blurred,
                     poor, read, driving, working, TV, windy, humidity, ac])

        if int(p[0]) == 0:
            pr = "Normal eye condition"
        elif int(p[0]) == 1:
            pr = "Mild dry eye condition"
        elif int(p[0]) == 2:
            pr = "Moderate dry eye condition"
        else:
            pr = "Severe dry eye condition"

        return render_template('result.html', prediction=pr)

    return render_template('index.html', prediction=None)


if __name__ == "__main__":
    app.run(debug=True)
