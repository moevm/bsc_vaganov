# This Python file uses the following encoding: utf-8
from data_collect_and_preprocess.leninka_scrapper import LeninkaScrapper as LS
from data_collect_and_preprocess.freq_analysis import TextFrequencyAnalyzer as TFA
from model.create_model import ModelHandler as MH

# TODO
# 1. Запуск из командной строки с задачей параметров dirpath и аргументов методов scrap(), decisionTreeCreate()
dirpath = 'papers/'
scrapper = LS()
if scrapper.scrap(250, 50, 1, 101):
    analyzer = TFA()
    if analyzer.getStats(dirpath, True):
        print("Data is ready.")
        model = MH()
        model.decisionTreeTest()
        model.decisionTreeCreate()
#model = MH()
#model.decisionTreeTest(max_depth=15, min_samples_split=20, min_samples_leaf=20)
#model.decisionTreeCreate(max_depth=15, min_samples_split=20, min_samples_leaf=20)
