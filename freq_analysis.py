import os
import codecs
import mistune
import csv
import argparse
import sys
from glob import glob
from pathlib import Path
from bs4 import BeautifulSoup
import re
import pymorphy2
import scipy.stats as ss
import math
import numpy
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import textract
import operator
import json
import nltk
from nltk.util import ngrams
from textblob import TextBlob

nltk.download('stopwords')


def CheckForDir(dirpath):
    dir = Path(dirpath)

    if dir.is_dir():
        print('Directory     ' + str(dir).ljust(25) + '...\tExists!')
        return True
    else:
        print('Directory     ' + str(dir).ljust(25) + "...\tDoesn't exist!")
        return False


def CheckForFile(dirpath, filename):
    file = Path(os.path.join(dirpath, filename))

    if file.is_file():
        print('File     ' + filename.ljust(25) + '...\tExists!')
        return True
    else:
        print('File     ' + filename.ljust(25) + "...\tDoesn't exist!")
        return False


def CheckForFileFullPath(filename):
    file = Path(os.path.abspath(filename))

    if file.is_file():
        print('File     ' + filename.ljust(25) + '...\tExists!')
        return True
    else:
        print('File     ' + filename.ljust(25) + "...\tDoesn't exist!")
        return False


def GetTextFromPdf(filename):
    try:
        text = textract.process(filename, encoding='utf-8', method='pdftotext', language='rus')
        return text
    except Exception:
        return None


def getAllTextFromPdf(filePath):
    print('Processing ' + filePath + ' ...\n')
    allText = ''
    bytesarr = GetTextFromPdf(filePath)
    if bytesarr == None:
        return ''
    allText += bytesarr.decode('utf-8-sig') + ' '
    # print(allText.encode('utf-8-sig'))
    return allText


def checkForFileFullPath(filename):
    file = Path(os.path.abspath(filename))
    if file.is_file():
        print('File     ' + filename.ljust(25) + '...\tExists!')
        return True
    else:
        print('File     ' + filename.ljust(25) + "...\tDoesn't exist!")
        return False


def getFilesFromFolder(dirpath):
    files = os.listdir(dirpath)
    pdfFiles = [f for f in files if f.lower().endswith(".pdf")]
    pdfFiles = [os.path.join(dirpath, f) for f in pdfFiles]
    return pdfFiles


def countWords(wordList):
    morph = pymorphy2.MorphAnalyzer()
    counts = {}
    normalFormWordList = []
    for _word in wordList:
        currForm = morph.parse(_word)[0]
        nounForm = currForm.inflect({'sing', 'nomn'})

        try:
            word = nounForm.word
        except:
            word = currForm.word

        normalFormWordList.append(word)

    ruStopWords = set(stopwords.words('russian'))
    enStopWords = set(stopwords.words('english'))

    filteredWordList = [_word for _word in normalFormWordList if _word not in ruStopWords]
    filteredWordList = [_word for _word in filteredWordList if _word not in enStopWords]
    filteredWordList = [_word for _word in filteredWordList if not _word.isdigit()]

    for _word in filteredWordList:
        if _word in counts:
            counts[_word] += 1
        else:
            counts[_word] = 1

    return counts

# Проверка наличия местоимений
def checkPronounce(wordList):
    score = 0
    morph = pymorphy2.MorphAnalyzer()
    for word in wordList:
        currForm = morph.parse(word)[0]
        if currForm.tag.POS == 'NPRO':
            score = score - 1
    print(score)
    return score



def checkWater(wordList):
    ruStopWords = set(stopwords.words('russian'))
    stopWords = [_word for _word in wordList if _word in ruStopWords]
    waterLevel = len(stopWords) / len(wordList) * 100
    return len(stopWords), waterLevel


def GetKeyWords(wordList):
    sortedWordList = sorted(wordList.items(), key=operator.itemgetter(1))[::-1]

    maxKey, maxValue = sortedWordList[0]
    keyWords = [(w, c) for (w, c) in sortedWordList if c >= maxValue / 2]
    return keyWords


def GetYPlot(data):
    _data = []
    _max = max(data)
    for i in range(0, len(data)):
        _data.append(_max / (i + 1))
    return _data


def GetStandartDeviation(data):
    maxElem = max(data)
    perfectData = []
    deviation = 0
    for i in range(0, len(data)):
        perfectData.append(data[0] / (i + 1))
    for i in range(0, len(data)):
        deviation += math.pow(data[i] - perfectData[i], 2)
    return math.sqrt(deviation / len(data))


def checkFormalRequirements(data, bigrams):
    haveAnnotation = 0
    haveConclusion = 0
    haveProblemFormulation = 0
    haveReferences = 0
    havePaperKeyWords = 0
    haveMethodDescription = 0
    for word in data:
        if haveAnnotation == 0 and word == "введение" or word == "аннотация":
            haveAnnotation = 1
        if haveConclusion == 0 and word == "выводы" or word == "заключение" or word == "результаты":
            haveConclusion = 1

    for bigram in list(bigrams):
        if haveProblemFormulation == 0 and bigram == ('постановка', 'задачи'):
            haveProblemFormulation = 1
        if haveReferences == 0 and bigram == ('список', 'литературы') or bigram == ('библиографический', 'список'):
            haveReferences = 1
        if havePaperKeyWords == 0 and bigram == ('ключевые', 'слова'):
            havePaperKeyWords = 1
        if haveMethodDescription == 0 and bigram == ('выбор', 'метода') or bigram == ('описание', 'метода'):
            haveMethodDescription = 1

    return haveMethodDescription + havePaperKeyWords + haveReferences + haveProblemFormulation + haveConclusion + haveAnnotation



def GetStats(dirPath, flagDir):
    if not flagDir:
        if not checkForFileFullPath(dirPath):
            #sys.exit()
            return

        allText = getAllTextFromPdf(dirPath)
        wordList = re.sub("[^\w]", " ", allText).split()

        wordList = [w.lower() for w in wordList]
        if len(wordList) == 0:
            print("No words in text!!!")
            #sys.exit()
            return
        #f = open('results.txt', 'w')
        water = checkWater(wordList)
        counts = countWords(wordList)
        keyWords = GetKeyWords(counts)


            # for word, freq in counts.items():
            #     if freq > 5:
            #         print(word + ": " + str(freq))

            # print("Keywords in text:")
            # for word, freq in keyWords:
            #     if freq > 5:
            #         print(word + ": " + str(freq))
        keyWordsLevel = sum([pair[1] for pair in keyWords]) / sum(counts.values())
        #keyl = 5.5
        #keyr = 13.5
        #waterl = 14
        #waterr = 20.5
        #devl = 5
        #devr = 9.5
        print("Keywords level: " + str(keyWordsLevel * 100) + "%")
        #if keyl <= keyWordsLevel * 100 <= keyr:
        #    print("Keywords level Good")
        #else:
        #       print("Keywords level Bad")

        print("Stopwords in text: " + str(water[0]))
        print("Waterlevel: " + str(water[1]) + "%")
        #if waterl <= water[1] <= waterr:
        #    print("Waterlevel level Good")
        #else:
        #    print("Waterlevel level Bad")

        #amb = [(w, c) for (w, c) in counts.items()]
        #amb_c_rank = ss.rankdata([c for (w, c) in amb])
        #amb_sorted = sorted(amb, key=lambda x: x[1], reverse=True)

        #x = range(0, len(amb_sorted[0:]))
        #y = [c for (w, c) in amb_sorted[0:]]
        #y2 = GetYPlot([c for (w, c) in amb_sorted[0:]])

        deviation = GetStandartDeviation([c for (w, c) in amb_sorted if c >= 5])  # GetStandartDeviation(y3)

        #f.write(str(
        #    {'filename': dirPath, 'keywordsLvl': keyWordsLevel * 100, 'WaterLvl': water[1], 'devition': deviation}))
        #print("deviation: " + str(deviation))

        #if devl <= deviation <= devr:
        #    print("deviation level Good")
        #else:
        #    print("deviation level Bad")

        #my_xticks = [w for (w, c) in amb_sorted[0:]]
        #plt.ylabel("Частота употребления слова")
        #plt.xlabel("Ранг частоты употребления слова")
        #plt.plot(x, y, color='k')
        #plt.plot(x, y2, ':', color='k')
        #plt.show()



    else:
        if not CheckForDir(dirPath):
            #sys.exit()
            return
        files = getFilesFromFolder(dirPath)
        # f = open('results.txt', 'w')
        files.sort()
        print(files)
        papers_freq = []
        for pdfFile in files:
            allText = getAllTextFromPdf(pdfFile)
            wordList = re.sub("[^\w]", " ", allText).split()
            bigramsList = ngrams(wordList, 2)
            wordList = [w.lower() for w in wordList]
            if len(wordList) == 0:
                continue
            pronounceScore = checkPronounce(wordList)
            sentiment_analysis = TextBlob(allText)
            water = checkWater(wordList)
            counts = countWords(wordList)
            print(counts)
            if counts == {}:
                continue
            keyWords = GetKeyWords(counts)
            # калькулятор ключевых биграмм, поиск названий разделов статьи
            keyWordsLevel = sum([pair[1] for pair in keyWords]) / sum(counts.values())

            amb = [(w, c) for (w, c) in counts.items()]
            amb_c_rank = ss.rankdata([c for (w, c) in amb])
            amb_sorted = sorted(amb, key=lambda x: x[1], reverse=True)

            #x = range(0, len(amb_sorted[0:]))
            #y = [c for (w, c) in amb_sorted[0:]]
            #y2 = GetYPlot([c for (w, c) in amb_sorted[0:]])

            data = [c for (w, c) in amb_sorted if c >= 5]
            if not data:
                continue
            else:
                deviation = GetStandartDeviation(data)  # GetStandartDeviation(y3)

            formalScore = checkFormalRequirements(wordList, bigramsList)

            paper_freq_data = {
                'paperPath': pdfFile,
                'keywordsLvl': keyWordsLevel * 100,
                'WaterLvl': water[1],
                'deviation': deviation,
                'polarity': sentiment_analysis.sentiment.polarity * 100,
                'subjectivity': sentiment_analysis.sentiment.subjectivity * 100,
                'formalScore': formalScore,
                'pronounceScore': pronounceScore,
                'length': len(allText),
                'lexicalDiversity': len(set(allText)) / len(allText) * 100
            }
            papers_freq.append(paper_freq_data)
            #f.write(str({'filename': pdfFile, 'keywordsLvl': keyWordsLevel * 100, 'WaterLvl': water[1],
            #             'devition': deviation}))
            #f.write(', ')
        with open('freqdata.json', 'w', encoding='utf-8') as f:
            json.dump(papers_freq, f, ensure_ascii=False)


GetStats('/home/woghan/Desktop/ml/leninka_scrapper/papers/', True)
