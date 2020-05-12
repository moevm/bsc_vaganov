from flask import Flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas
from data_collect_and_preprocess.freq_analysis import TextFrequencyAnalyzer as TFA
app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')