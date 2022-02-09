#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('credit_model.pkl', 'rb'))
scaler =pickle.load(open('scaler.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('template_credit.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        '''
    For rendering results on HTML GUI
    '''
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        final_features=scaler.transform(final_features)
    prediction = model.predict(final_features)
    if prediction == 1:
        
        output="LOAN APPROVED"
    else:
        output='LOAN APPLICATION REJECTED'
    

    return render_template('template_credit.html', prediction_text= output)


if __name__ == "__main__":
    app.run(debug=True)
