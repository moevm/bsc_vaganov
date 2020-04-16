import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from data_collect_and_preprocess.freq_analysis import TextFrequencyAnalyzer as TFA

app = Flask(__name__)
model = pickle.load(open('../model/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    analysis = TFA()
    paper_obj = analysis.analyseSinglePaper(request.form['text'])
    prediction = model.predict(paper_obj)
    print(prediction)
    if str(prediction) == '[0]':
        return render_template('index.html', prediction_text='Your paper is bad')
    else:
        return render_template('index.html', prediction_text='Your paper is good')


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
