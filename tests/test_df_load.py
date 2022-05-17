from context import aiwf, test_data_1, test_data_2, test_data_3, test_data_4
import pandas as pd
import unittest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        df = test_data_4
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
        test_case_4 = pd.DataFrame(np.array(ct.fit_transform(df)), columns=['State 1', 'State 2', 'State 3', 'R&D Spend', 'Administration', 'Marketing spend',  'Profit'])
        test_case_2 = test_data_2
        test_case_3 = test_data_3
        test_case_3.pop('Position')

        self.assertIsNone(aiwf.run_wf(test_case_4))


if __name__ == '__main__':
    unittest.main()
