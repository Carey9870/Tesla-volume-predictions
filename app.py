from flask import Flask, app, url_for, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd

# instantiation
app = Flask(__name__)

# load the pickle model file
reg = pickle.load(open('regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output = reg.predict(final_input)
    return render_template('home.html', prediction_text="The Tesla predicted Volume is {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)