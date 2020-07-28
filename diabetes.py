
import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model= load('randomforest2.save')
trans=load('standardscaler2')

@app.route('/')
def home():
    return render_template('html_page.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    test=trans.transform(x_test)
    prediction = model.predict(test)
    print(prediction)
    output=prediction[0]
    if output==0:
        return render_template('precautions.html', prediction_text='The individual does not have diabetes')
    else:
        return render_template('advice.html', prediction_text='The individual has  diabetes')
        

if __name__ == "__main__":
    app.run(debug=True)