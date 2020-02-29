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


def checkWater(wordList):
    ruStopWords = set(stopwords.words('russian'))
    stopWords = [_word for _word in wordList if _word in ruStopWords]
    waterLevel = len(stopWords) / len(wordList) * 100
    return (len(stopWords), waterLevel)


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


def GetStats(dirPath, flagDir):
    if not flagDir:
        if not checkForFileFullPath(dirPath):
            #sys.exit()
            return

        try:
            allText = getAllTextFromPdf(dirPath)
            wordList = re.sub("[^\w]", " ", allText).split()
            wordList = [w.lower() for w in wordList]
            if len(wordList) == 0:
                print("No words in text!!!")
                #sys.exit()
                return
            f = open('results.txt', 'w')
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
            keyl = 5.5
            keyr = 13.5
            waterl = 14
            waterr = 20.5
            devl = 5
            devr = 9.5
            print("Keywords level: " + str(keyWordsLevel * 100) + "%")
            if keyl <= keyWordsLevel * 100 <= keyr:
                print("Keywords level Good")
            else:
                print("Keywords level Bad")

            print("Stopwords in text: " + str(water[0]))
            print("Waterlevel: " + str(water[1]) + "%")
            if waterl <= water[1] <= waterr:
                print("Waterlevel level Good")
            else:
                print("Waterlevel level Bad")

            amb = [(w, c) for (w, c) in counts.items()]
            amb_c_rank = ss.rankdata([c for (w, c) in amb])
            amb_sorted = sorted(amb, key=lambda x: x[1], reverse=True)

            x = range(0, len(amb_sorted[0:]))
            y = [c for (w, c) in amb_sorted[0:]]
            y2 = GetYPlot([c for (w, c) in amb_sorted[0:]])

            deviation = GetStandartDeviation([c for (w, c) in amb_sorted if c >= 5])  # GetStandartDeviation(y3)

            f.write(str(
                {'filename': dirPath, 'keywordsLvl': keyWordsLevel * 100, 'WaterLvl': water[1], 'devition': deviation}))
            print("deviation: " + str(deviation))

            if devl <= deviation <= devr:
                print("deviation level Good")
            else:
                print("deviation level Bad")

            my_xticks = [w for (w, c) in amb_sorted[0:]]
            plt.ylabel("Частота употребления слова")
            plt.xlabel("Ранг частоты употребления слова")
            plt.plot(x, y, color='k')
            plt.plot(x, y2, ':', color='k')
            plt.show()
        except Exception:
            #sys.exit()
            return
    else:
        if not CheckForDir(dirPath):
            #sys.exit()
            return
        files = getFilesFromFolder(dirPath)
        f = open('results.txt', 'w')
        files.sort()
        print(files)
        for pdfFile in files:
            try:
                allText = getAllTextFromPdf(pdfFile)
                wordList = re.sub("[^\w]", " ", allText).split()
                wordList = [w.lower() for w in wordList]
                if len(wordList) == 0:
                    continue
                water = checkWater(wordList)
                counts = countWords(wordList)
                keyWords = GetKeyWords(counts)

                keyWordsLevel = sum([pair[1] for pair in keyWords]) / sum(counts.values())

                amb = [(w, c) for (w, c) in counts.items()]
                amb_c_rank = ss.rankdata([c for (w, c) in amb])
                amb_sorted = sorted(amb, key=lambda x: x[1], reverse=True)

                x = range(0, len(amb_sorted[0:]))
                y = [c for (w, c) in amb_sorted[0:]]
                y2 = GetYPlot([c for (w, c) in amb_sorted[0:]])

                deviation = GetStandartDeviation([c for (w, c) in amb_sorted if c >= 5])  # GetStandartDeviation(y3)

                f.write(str({'filename': pdfFile, 'keywordsLvl': keyWordsLevel * 100, 'WaterLvl': water[1],
                             'devition': deviation}))
                f.write(', ')
            except Exception:
                continue
        f.close()


for i in range(1, 112):
    GetStats('D:/Users/nikva/PyCharmProjects/ml/leninka_scrapper/papers/paper' + str(i) + '.pdf', False)
