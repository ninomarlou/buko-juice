from context import aiwf, test_data_simple_linear_regression, test_data_2, test_data_3, test_data_multiple_linear_regression, test_data_polynomial_regression
import pandas as pd
import unittest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        # simple linear regression sample
        test_case_1 = test_data_simple_linear_regression

        # multiple linear regression sample example
        df = test_data_multiple_linear_regression
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
        test_case_2 = pd.DataFrame(np.array(ct.fit_transform(df)), columns=[
                                   'State 1', 'State 2', 'State 3', 'R&D Spend', 'Administration', 'Marketing spend',  'Profit'])

        # polynomial regression sample
        test_case_3 = test_data_polynomial_regression
        test_case_3.pop('Position')

        print('Test cases:')
        print('[ 1 ] Simple linear regression')
        print('[ 2 ] Multiple linear regression')
        print('[ 3 ] Polynomial regression')
        print('[ 4 ] Support vector regression')
        print('[ 5 ] Decision tree regression')
        print('[ 6 ] Random forest regression')

        result = int(input('Which test do you want to run? '))

        if result == 1:
            test_case = test_case_1
        elif result == 2:
            test_case = test_case_2
        elif result in (3, 4, 5, 6):
            test_case = test_case_3
        else:
            exit

        self.assertIsNone(aiwf.run_wf(test_case))


if __name__ == '__main__':
    unittest.main()
