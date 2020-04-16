# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import re
import pymorphy2
import math
from nltk.corpus import stopwords
import textract
import operator
import json
import nltk
import numpy as np
from nltk.util import ngrams
from textblob import TextBlob

nltk.download('stopwords')


class TextFrequencyAnalyzer:

    def checkForDir(self, dirpath):
        dir_path = Path(dirpath)
        if dir_path.is_dir():
            print('Directory     ' + str(dir_path).ljust(25) + '...\tExists!')
            return True
        else:
            print('Directory     ' + str(dir_path).ljust(25) + "...\tDoesn't exist!")
            return False

    def checkForFile(self, dirpath, filename):
        file = Path(os.path.join(dirpath, filename))
        if file.is_file():
            print('File     ' + filename.ljust(25) + '...\tExists!')
            return True
        else:
            print('File     ' + filename.ljust(25) + "...\tDoesn't exist!")
            return False

    def checkForFileFullPath(self, filename):
        file = Path(os.path.abspath(filename))
        if file.is_file():
            print('File     ' + filename.ljust(25) + '...\tExists!')
            return True
        else:
            print('File     ' + filename.ljust(25) + "...\tDoesn't exist!")
            return False

    def getTextFromPdf(self, filename):
        try:
            text = textract.process(filename, encoding='utf-8', method='pdftotext', language='rus')
            return text
        except Exception:
            return None

    def getAllTextFromPdf(self, filePath):
        print('Processing ' + filePath + ' ...\n')
        allText = ''
        bytesarr = self.getTextFromPdf(filePath)
        if bytesarr is None:
            return ''
        allText += bytesarr.decode('utf-8-sig') + ' '
        # print(allText.encode('utf-8-sig'))
        return allText

    def getFilesFromFolder(self, dirpath):
        files = os.listdir(dirpath)
        pdfFiles = [f for f in files if f.lower().endswith(".pdf")]
        pdfFiles = [os.path.join(dirpath, f) for f in pdfFiles]
        return pdfFiles

    def countWords(self, wordList):
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

    def checkPronounce(self, wordList):
        score = 0
        morph = pymorphy2.MorphAnalyzer()
        for word in wordList:
            currForm = morph.parse(word)[0]
            if currForm.tag.POS == 'NPRO':
                score = score - 1
        print(score)
        return score

    def checkWater(self, wordList):
        ruStopWords = set(stopwords.words('russian'))
        stopWords = [_word for _word in wordList if _word in ruStopWords]
        waterLevel = len(stopWords) / len(wordList) * 100
        return len(stopWords), waterLevel

    def getKeyWords(self, wordList):
        sortedWordList = sorted(wordList.items(), key=operator.itemgetter(1))[::-1]

        maxKey, maxValue = sortedWordList[0]
        keyWords = [(w, c) for (w, c) in sortedWordList if c >= maxValue / 2]
        return keyWords

    def getStandartDeviation(self, data):
        maxElem = max(data)
        perfectData = []
        deviation = 0
        for i in range(0, len(data)):
            perfectData.append(data[0] / (i + 1))
        for i in range(0, len(data)):
            deviation += math.pow(data[i] - perfectData[i], 2)
        return math.sqrt(deviation / len(data))

    def checkFormalRequirements(self, data, bigrams):
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

    def analyseSinglePaper(self, text):
        wordList = re.sub("[^\w]", " ", text).split()
        bigramsList = ngrams(wordList, 2)
        wordList = [w.lower() for w in wordList]
        if len(wordList) == 0:
            return
        pronounceScore = self.checkPronounce(wordList)
        sentiment_analysis = TextBlob(text)
        water = self.checkWater(wordList)
        counts = self.countWords(wordList)
        if counts == {}:
            return False
        keyWords = self.getKeyWords(counts)
        # калькулятор ключевых биграмм, поиск названий разделов статьи
        keyWordsLevel = sum([pair[1] for pair in keyWords]) / sum(counts.values())

        amb = [(w, c) for (w, c) in counts.items()]
        amb_sorted = sorted(amb, key=lambda x: x[1], reverse=True)

        data = [c for (w, c) in amb_sorted if c >= 5]
        if not data:
            return False
        else:
            deviation = self.getStandartDeviation(data)  # GetStandartDeviation(y3)

        formalScore = self.checkFormalRequirements(wordList, bigramsList)

        paper_freq_data = [
            keyWordsLevel * 100,
            water[1],
            deviation,
            sentiment_analysis.sentiment.polarity * 100,
            sentiment_analysis.sentiment.subjectivity * 100,
            formalScore,
            pronounceScore,
            len(text),
            len(set(text)) / len(text) * 100
        ]
        return [np.array(paper_freq_data)]

    def getStats(self, dirPath, flagDir):
        if not flagDir:
            if not self.checkForFileFullPath(dirPath):
                #sys.exit()
                return False

            allText = self.getAllTextFromPdf(dirPath)
            wordList = re.sub("[^\w]", " ", allText).split()
            wordList = [w.lower() for w in wordList]
            if len(wordList) == 0:
                print("No words in text")
                #sys.exit()
                return False
            water = self.checkWater(wordList)
            counts = self.countWords(wordList)
            keyWords = self.getKeyWords(counts)
            keyWordsLevel = sum([pair[1] for pair in keyWords]) / sum(counts.values())
            print("Keywords level: " + str(keyWordsLevel * 100) + "%")
            print("Stopwords in text: " + str(water[0]))
            print("Waterlevel: " + str(water[1]) + "%")
        else:
            if not self.checkForDir(dirPath):
                #sys.exit()
                return False
            files = self.getFilesFromFolder(dirPath)
            files.sort()
            print(files)
            papers_freq = []
            for pdfFile in files:
                allText = self.getAllTextFromPdf(pdfFile)
                wordList = re.sub("[^\w]", " ", allText).split()
                bigramsList = ngrams(wordList, 2)
                wordList = [w.lower() for w in wordList]
                if len(wordList) == 0:
                    continue
                pronounceScore = self.checkPronounce(wordList)
                sentiment_analysis = TextBlob(allText)
                water = self.checkWater(wordList)
                counts = self.countWords(wordList)
                print(counts)
                if counts == {}:
                    continue
                keyWords = self.getKeyWords(counts)
                # калькулятор ключевых биграмм, поиск названий разделов статьи
                keyWordsLevel = sum([pair[1] for pair in keyWords]) / sum(counts.values())
                amb = [(w, c) for (w, c) in counts.items()]
                amb_sorted = sorted(amb, key=lambda x: x[1], reverse=True)
                data = [c for (w, c) in amb_sorted if c >= 5]
                if not data:
                    continue
                else:
                    deviation = self.getStandartDeviation(data)  # GetStandartDeviation(y3)
                formalScore = self.checkFormalRequirements(wordList, bigramsList)

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
            with open('freqdata.json', 'w', encoding='utf-8') as f:
                json.dump(papers_freq, f, ensure_ascii=False)


#getStats('/home/woghan/Desktop/ml/bsc_vaganov/papers/', True)
