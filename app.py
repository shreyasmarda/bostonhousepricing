import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy
import pandas
# from django.shortcuts import render
app = Flask(__name__)

## Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    # print(data)
    # print(numpy.array(list(data.values())).reshape(1, -1))
    # print(numpy.array(list(data.values())).reshape(1, -1).shape)
    new_data = scalar.transform(numpy.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    # print(output[0])
    return jsonify(output[0])
    # return list([0])

if __name__== "__main__":
    app.run(debug = True)
