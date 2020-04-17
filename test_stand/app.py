import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from data_collect_and_preprocess.freq_analysis import TextFrequencyAnalyzer as TFA

app = Flask(__name__)
model = pickle.load(open('../model/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        analysis = TFA()
        paper_obj = analysis.analyseSinglePaper(request.form['text'], False)
        prediction = model.predict(paper_obj)
        print(prediction)
        if str(prediction) == '[0]':
            return render_template('predict_text.html', prediction_text='Your paper is bad')
        else:
            return render_template('predict_text.html', prediction_text='Your paper is good')
    elif request.method == 'GET':
        return render_template('predict_text.html')


@app.route('/predict_pdf', methods=['POST', 'GET'])
def predict_pdf():
    if request.method == 'POST':
        analysis = TFA()
        f = request.files['file']
        f.save('/home/woghan/Desktop/bsc_vaganov/test_stand/samples/test.pdf')
        paper_obj = analysis.analyseSinglePaper('/home/woghan/Desktop/bsc_vaganov/test_stand/samples/test.pdf', True)
        prediction = model.predict(paper_obj)
        print(prediction)
        if str(prediction) == '[0]':
            return render_template('predict_file.html', prediction_text='Your paper is bad')
        else:
            return render_template('predict_file.html', prediction_text='Your paper is good')
    elif request.method == 'GET':
        return render_template('predict_file.html')


if __name__ == "__main__":
    app.run(debug=True)
