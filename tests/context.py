

import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import aiwf
test_data_1 = pd.read_csv(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')) + '/tests/test_data_1.csv')
test_data_2 = pd.read_csv(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')) + '/tests/test_data_2.csv')
test_data_3 = pd.read_csv(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')) + '/tests/test_data_3.csv')
test_data_4 = pd.read_csv(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')) + '/tests/test_data_4.csv')
