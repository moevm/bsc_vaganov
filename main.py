# This Python file uses the following encoding: utf-8
from data_collect_and_preprocess.leninka_scrapper import LeninkaScrapper as LS
from data_collect_and_preprocess.freq_analysis import TextFrequencyAnalyzer as TFA
from model.create_model import ModelHandler as MH

# TODO
# 1. Запуск из командной строки с задачей параметров dirpath и аргументов методов scrap(), decisionTreeCreate()
dirpath = '/home/woghan/Desktop/ml/bsc_vaganov/papers/'
scrapper = LS()
if scrapper.scrap(150, 20, 50):
    analyzer = TFA()
    if analyzer.getStats(dirpath, True):
        print("Data is ready.")
        model = MH()
        model.decisionTreeCreate()


