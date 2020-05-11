import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas
from data_collect_and_preprocess.freq_analysis import TextFrequencyAnalyzer as TFA

app = Flask(__name__)
model = pickle.load(open('../model/model.pkl', 'rb'))
#mean values
keywordsLvl = 10.85
waterLvl = 17.08
deviation = 6.45
polarity = -2.03
subjectivity = 35.71
formalScore = 0.95
pronounceScore = -20.15
length = 14616
lexicalDiversity = 1.35


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/data')
def data():
    return render_template('data.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        if len(request.form['text']) < 1000:
            return render_template('predict_text.html', error_message='Введенный текст слишком короткий, невозможно классифицировать')
        analysis = TFA()
        paper_obj = analysis.analyseSinglePaper(request.form['text'], False)
        prediction = model.predict(paper_obj)
        proba = model.predict_proba(paper_obj)
        recommendations = []
        print(paper_obj)
        if 5.5 <= paper_obj[0][0] <= 13.5:
            #good
            recommendations.append("Хороший уровень ключевых слов, соответствует научному стилю")
        else:
            #bad
            recommendations.append("Плохой уровень ключевых слов (слишком много или слишком мало)")
        if 14 <= paper_obj[0][1] <= 20.5:
            # good
            recommendations.append("Хороший уровень водности текста, соответствует научному стилю.")
        else:
            # bad
            recommendations.append("Плохой уровень водности текста (слишком много или слишком мало повторов ключевых слов в тексте)")
        if 5 <= paper_obj[0][2] <= 9.5:
            # good
            recommendations.append("Текст выглядит естественно.")
        else:
            # bad
            recommendations.append("Текст выглядит неестественно.")
        if -20 <= paper_obj[0][6] <= -50:
            # good
            recommendations.append("В тексте не слишком много личных местоимений.")
        else:
            # bad
            recommendations.append("В тексте слишком много или слишком мало личных местоимений")
        if 8000 <= paper_obj[0][7] <= 22000:
            # good
            recommendations.append("Размер текста не слишком большой, но и не слишком маленький.")
        else:
            # bad
            recommendations.append("Текст слишком большой или слишком маленький.")

        print(prediction)
        if str(prediction) == '[0]':
            return render_template('predict_text.html', prediction_negative=round(proba[0][0], 3), recommends=recommendations)
        else:
            return render_template('predict_text.html', prediction_positive=round(proba[0][1], 3), recommends=recommendations)
    elif request.method == 'GET':
        return render_template('predict_text.html')


@app.route('/predict_pdf', methods=['POST', 'GET'])
def predict_pdf():
    if request.method == 'POST':
        analysis = TFA()
        f = request.files['file']
        if f.filename == '':
            return render_template('predict_file.html', error_message='Невозможно прочитать файл.')
        f.save('/home/woghan/Desktop/bsc_vaganov/test_stand/samples/test.pdf')
        paper_obj = analysis.analyseSinglePaper('/home/woghan/Desktop/bsc_vaganov/test_stand/samples/test.pdf', True)
        prediction = model.predict(paper_obj)
        proba = model.predict_proba(paper_obj)
        recommendations = []
        print(paper_obj)
        if 5.5 <= paper_obj[0][0] <= 13.5:
            # good
            recommendations.append("Хороший уровень ключевых слов, соответствует научному стилю")
        else:
            # bad
            recommendations.append("Плохой уровень ключевых слов (слишком много или слишком мало)")
        if 14 <= paper_obj[0][1] <= 20.5:
            # good
            recommendations.append("Хороший уровень водности текста, соответствует научному стилю.")
        else:
            # bad
            recommendations.append(
                "Плохой уровень водности текста (слишком много или слишком мало повторов ключевых слов в тексте)")
        if 5 <= paper_obj[0][2] <= 9.5:
            # good
            recommendations.append("Текст выглядит естественно.")
        else:
            # bad
            recommendations.append("Текст выглядит неестественно.")
        if -20 <= paper_obj[0][6] <= -50:
            # good
            recommendations.append("В тексте не слишком много личных местоимений.")
        else:
            # bad
            recommendations.append("В тексте слишком много или слишком мало личных местоимений")
        if 8000 <= paper_obj[0][7] <= 22000:
            # good
            recommendations.append("Размер текста не слишком большой, но и не слишком маленький.")
        else:
            # bad
            recommendations.append("Текст слишком большой или слишком маленький.")
        if str(prediction) == '[0]':
            return render_template('predict_file.html', prediction_negative=round(proba[0][0], 3), recommends=recommendations)
        else:
            return render_template('predict_file.html', prediction_positive=round(proba[0][1], 3), recommends=recommendations)
    elif request.method == 'GET':
        return render_template('predict_file.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
