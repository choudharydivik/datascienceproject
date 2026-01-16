from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            input_df = pd.DataFrame([{
                "fixed acidity": fixed_acidity,
                "volatile acidity": volatile_acidity,
                "citric acid": citric_acid,
                "residual sugar": residual_sugar,
                "chlorides": chlorides,
                "free sulfur dioxide": free_sulfur_dioxide,
                "total sulfur dioxide": total_sulfur_dioxide,
                "density": density,
                "pH": pH,
                "sulphates": sulphates,
                "alcohol": alcohol
            }])

            obj = PredictionPipeline()
            pred = obj.predict(input_df)
            final_prediction = float(pred[0])

            return render_template('results.html', prediction=final_prediction)

        except Exception as e:
            print('The Exception message is:', e)
            return render_template('results.html', prediction=None)

    return render_template('index.html')



if __name__ == "__main__":
	
	app.run(host="0.0.0.0", port = 8080)