#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('template.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features=scaler.transform(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 3)

    return render_template('template.html', prediction_text='Estimated value of property is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    model.feature_importances_
