import pandas as pd
from .context import localisation
from .regression import Regression
from .classification import Classification


def run_wf(df):
    # Check if variable received is a pandas dataframe
    if isinstance(df, pd.DataFrame) == False:
        raise TypeError(localisation.messages.df_type_error)
        exit

    # Prompt user to select column to use as label.
    df_columns = list(df.columns)
    print(localisation.messages.df_columns_list_message)
    for i in range(0, len(df_columns), 1):
        print('   [', i + 1, ']', df_columns[i])

    df_column_label_index = user_input_with_validation(localisation.messages.df_select_label_message, range(1, len(df_columns) + 1), 1)
    # Create feature and label dataframes.
    y = df[df_columns[df_column_label_index]]
    X = df.copy()
    X.pop(df_columns[df_column_label_index])

    # Ask for test test_size
    test_size = user_input_with_validation(localisation.messages.range_message, range(0, 2), 2)

    print(localisation.messages.problem_type_message)
    for i in range(0, len(localisation.messages.problem_types), 1):
        print('   [', i + 1, ']', localisation.messages.problem_types[i])
    problem_type = user_input_with_validation(localisation.messages.problem_type_query, range(1, len(localisation.messages.problem_types) + 1), 1)

    if localisation.messages.problem_types[problem_type] == 'Regression':
        model = Regression(test_size, X.values, y.values)
    else:
        model = Classification(test_size, X.values, y.values)

    model.run()


def user_input_with_validation(message, response_range, range_type):
    block = 1
    if range_type == 1:
        # Discrete range
        while block == 1:
            result = input('   ' + message + ' ')
            try:
                result = int(result)
                if result in response_range:
                    return int(result) - 1
                else:
                    print('  ', localisation.messages.select_sorry_response + ',', result, localisation.messages.select_response_not_in_range)
            except ValueError:
                print('  ', localisation.messages.select_sorry_response + ',', result, localisation.messages.select_response_wrong_type)
                pass  # it was a string, not an int.
    if range_type == 2:
        # Continous range_type
        while block == 1:
            result = input(message + ' ')
            try:
                result = float(result)
                if result >= min(response_range) and result <= max(response_range):
                    return float(result)
                else:
                    print('  ', localisation.messages.select_sorry_response + ',', result, localisation.messages.select_response_not_in_range)
            except ValueError:
                print('  ', localisation.messages.select_sorry_response + ',', result, localisation.messages.select_response_wrong_type)
                pass  # it was a string, not an int.
