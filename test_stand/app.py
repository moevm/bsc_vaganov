import pickle
from flask import Flask, request, render_template
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
#extreme values
keywordsLvl_left = 5.5
keywordsLvl_right = 13.5
waterLvl_left = 14
waterLvl_right = 20.5
deviation_left = 5
deviation_right = 9.5
pronounceScore_left = -20
pronounceScore_right = -50
length_left = 8000
length_right = 22000

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
        if keywordsLvl_left <= paper_obj[0][0] <= keywordsLvl_right:
            # good
            recommendations.append({
                'criteria': "Уровень ключевых слов",
                'text': "Хороший уровень ключевых слов, соответствует научному стилю",
                'lvl': round(paper_obj[0][0], 3),
                'range': "[{},{}]".format(keywordsLvl_left, keywordsLvl_right),
                'result': True
            })
        elif paper_obj[0][0] < keywordsLvl_left:
            # bad
            recommendations.append({
                'criteria': "Уровень ключевых слов",
                'text': "Мало ключевых слов, не соответствует научному стилю",
                'lvl': round(paper_obj[0][0], 3),
                'range': "[{},{}]".format(keywordsLvl_left, keywordsLvl_right),
                'result': False
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Уровень ключевых слов",
                'text': "Текст перенасыщен ключевыми словами, попробуйте использовать больше синонимов",
                'lvl': round(paper_obj[0][0], 3),
                'range': "[{},{}]".format(keywordsLvl_left, keywordsLvl_right),
                'result': False
            })
        if waterLvl_left <= paper_obj[0][1] <= waterLvl_right:
            # good
            recommendations.append({
                'criteria': "Водность текста",
                'text': "Хороший уровень водности, соответствует научному стилю",
                'lvl': round(paper_obj[0][1], 3),
                'range': "[{},{}]".format(waterLvl_left, waterLvl_right),
                'result': True
            })
        elif paper_obj[0][1] < waterLvl_left:
            # bad
            recommendations.append({
                'criteria': "Водность текста",
                'text': "Низкий уровень водности - текст слишком сухой (как досье)",
                'lvl': round(paper_obj[0][1], 3),
                'range': "[{},{}]".format(waterLvl_left, waterLvl_right),
                'result': False
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Водность текста",
                'text': "Высокий уровень водности - текст содержит много слов, не несущих смысловой нагрузки",
                'lvl': round(paper_obj[0][1], 3),
                'range': "[{},{}]".format(waterLvl_left, waterLvl_right),
                'result': False
            })
        if deviation_left <= paper_obj[0][2] <= deviation_right:
            # good
            recommendations.append({
                'criteria': "Естественность текста",
                'text': "Текст выглядит естественно",
                'lvl': round(paper_obj[0][2], 3),
                'range': "[{},{}]".format(deviation_left, deviation_right),
                'result': True
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Естественность текста",
                'text': "Текст выглядит неестественно",
                'lvl': round(paper_obj[0][2], 3),
                'range': "[{},{}]".format(deviation_left, deviation_right),
                'result': False
            })
        if pronounceScore_right <= paper_obj[0][6] <= pronounceScore_left:
            # good
            recommendations.append({
                'criteria': "Уровень личных местоимений",
                'text': "В тексте не слишком много и не слишком мало личных местоимений",
                'lvl': round(paper_obj[0][6], 3),
                'range': "[{},{}]".format(pronounceScore_right, pronounceScore_left),
                'result': True
            })
        elif paper_obj[0][6] < pronounceScore_right:
            # bad
            recommendations.append({
                'criteria': "Уровень личных местоимений",
                'text': "В тексте слишком много личных местоимений. Попробуйте их убрать",
                'lvl': round(paper_obj[0][6], 3),
                'range': "[{},{}]".format(pronounceScore_right, pronounceScore_left),
                'result': False
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Уровень личных местоимений",
                'text': "В тексте мало личных местоимений. Отсутствие местоимений делает текст сухим и трудночитаемым",
                'lvl': round(paper_obj[0][6], 3),
                'range': "[{},{}]".format(pronounceScore_right, pronounceScore_left),
                'result': False
            })
        if length_left <= paper_obj[0][7] <= length_right:
            # good
            recommendations.append({
                'criteria': "Длина текста",
                'text': "Хороший размер текста",
                'lvl': round(paper_obj[0][7], 3),
                'range': "[{},{}]".format(length_left, length_right),
                'result': True
            })
        elif paper_obj[0][7] < length_left:
            # bad
            recommendations.append({
                'criteria': "Длина текста",
                'text': "Статья слишком короткая. Попробуйте увеличить ее размер.",
                'lvl': round(paper_obj[0][7], 3),
                'range': "[{},{}]".format(length_left, length_right),
                'result': False
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Длина текста",
                'text': "Статья слишком длинная. Попробуйте уменьшить ее размер.",
                'lvl': round(paper_obj[0][7], 3),
                'range': "[ {} ; {} ]".format(length_left, length_right),
                'result': False
            })
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
        if keywordsLvl_left <= paper_obj[0][0] <= keywordsLvl_right:
            # good
            recommendations.append({
                'criteria': "Уровень ключевых слов",
                'text': "Хороший уровень ключевых слов, соответствует научному стилю",
                'lvl': round(paper_obj[0][0], 3),
                'range': "[{},{}]".format(keywordsLvl_left, keywordsLvl_right),
                'result': True
            })
        elif paper_obj[0][0] < keywordsLvl_left:
            # bad
            recommendations.append({
                'criteria': "Уровень ключевых слов",
                'text': "Мало ключевых слов, не соответствует научному стилю",
                'lvl': round(paper_obj[0][0], 3),
                'range': "[{},{}]".format(keywordsLvl_left, keywordsLvl_right),
                'result': False
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Уровень ключевых слов",
                'text': "Текст перенасыщен ключевыми словами, попробуйте использовать больше синонимов",
                'lvl': round(paper_obj[0][0], 3),
                'range': "[{},{}]".format(keywordsLvl_left, keywordsLvl_right),
                'result': False
            })
        if waterLvl_left <= paper_obj[0][1] <= waterLvl_right:
            # good
            recommendations.append({
                'criteria': "Водность текста",
                'text': "Хороший уровень водности, соответствует научному стилю",
                'lvl': round(paper_obj[0][1], 3),
                'range': "[{},{}]".format(waterLvl_left, waterLvl_right),
                'result': True
            })
        elif paper_obj[0][1] < waterLvl_left:
            # bad
            recommendations.append({
                'criteria': "Водность текста",
                'text': "Низкий уровень водности - текст слишком сухой (как досье)",
                'lvl': round(paper_obj[0][1], 3),
                'range': "[{},{}]".format(waterLvl_left, waterLvl_right),
                'result': False
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Водность текста",
                'text': "Высокий уровень водности - текст содержит много слов, не несущих смысловой нагрузки",
                'lvl': round(paper_obj[0][1], 3),
                'range': "[{},{}]".format(waterLvl_left, waterLvl_right),
                'result': False
            })
        if deviation_left <= paper_obj[0][2] <= deviation_right:
            # good
            recommendations.append({
                'criteria': "Естественность текста",
                'text': "Текст выглядит естественно",
                'lvl': round(paper_obj[0][2], 3),
                'range': "[{},{}]".format(deviation_left, deviation_right),
                'result': True
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Естественность текста",
                'text': "Текст выглядит неестественно",
                'lvl': round(paper_obj[0][2], 3),
                'range': "[{},{}]".format(deviation_left, deviation_right),
                'result': False
            })
        if pronounceScore_right <= paper_obj[0][6] <= pronounceScore_left:
            # good
            recommendations.append({
                'criteria': "Уровень личных местоимений",
                'text': "В тексте не слишком много и не слишком мало личных местоимений",
                'lvl': round(paper_obj[0][6], 3),
                'range': "[{},{}]".format(pronounceScore_right, pronounceScore_left),
                'result': True
            })
        elif paper_obj[0][6] < pronounceScore_right:
            # bad
            recommendations.append({
                'criteria': "Уровень личных местоимений",
                'text': "В тексте слишком много личных местоимений. Попробуйте их убрать",
                'lvl': round(paper_obj[0][6], 3),
                'range': "[{},{}]".format(pronounceScore_right, pronounceScore_left),
                'result': False
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Уровень личных местоимений",
                'text': "В тексте мало личных местоимений. Отсутствие местоимений делает текст сухим и трудночитаемым",
                'lvl': round(paper_obj[0][6], 3),
                'range': "[{},{}]".format(pronounceScore_right, pronounceScore_left),
                'result': False
            })
        if length_left <= paper_obj[0][7] <= length_right:
            # good
            recommendations.append({
                'criteria': "Длина текста",
                'text': "Хороший размер текста",
                'lvl': round(paper_obj[0][7], 3),
                'range': "[{},{}]".format(length_left, length_right),
                'result': True
            })
        elif paper_obj[0][7] < length_left:
            # bad
            recommendations.append({
                'criteria': "Длина текста",
                'text': "Статья слишком короткая. Попробуйте увеличить ее размер.",
                'lvl': round(paper_obj[0][7], 3),
                'range': "[{},{}]".format(length_left, length_right),
                'result': False
            })
        else:
            # bad
            recommendations.append({
                'criteria': "Длина текста",
                'text': "Статья слишком длинная. Попробуйте уменьшить ее размер.",
                'lvl': round(paper_obj[0][7], 3),
                'range': "[ {} ; {} ]".format(length_left, length_right),
                'result': False
            })
        if str(prediction) == '[0]':
            return render_template('predict_file.html', prediction_negative=round(proba[0][0], 3), recommends=recommendations)
        else:
            return render_template('predict_file.html', prediction_positive=round(proba[0][1], 3), recommends=recommendations)
    elif request.method == 'GET':
        return render_template('predict_file.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
