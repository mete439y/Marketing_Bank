from flask import Flask, render_template, request, jsonify
import pickle, joblib
import json
import numpy as np 
import pandas as np
app = Flask(__name__)
## loaded the model
model= pickle.load( open("Rand_clf.pkl", "rb"))

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    input = [float(x) for x in request.form.values()]
    features = [np.array(input)]
    #print(input)
    output = model.predict(features)

    return render_template('index.html', prediction_text = 'Bank subscription prediction is {}'.format(output))
    
    
if __name__ == '__main__':
    app.run()