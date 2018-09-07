# Created by mqgao at 2018/8/31

"""
Feature: #Enter feature name here
# Enter feature description here

Scenario: #Enter scenario name here
# Enter steps here

Test File Location: # Enter

"""

from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os
from comment_predicate import test_model
from tools.convert_category import convert_continuous_to_n_categorical, change_n_categorical_to_n_binary_categorical

validation_csv_fpath = os.path.join('data', 'validation_binary.csv')
validation_original = pd.read_csv(validation_csv_fpath)
label_columns = validation_original.columns.tolist()
label_columns.remove('comment')

assert len(label_columns) == 80


def evalution(predicate_csv_content, true_csv_content, labeld_columns):
    return np.mean([f1_score(predicate_csv_content[c], true_csv_content[c]) for c in labeld_columns])


def get_validation_set_f1_score(model=None):
    validation_predicate = change_n_categorical_to_n_binary_categorical(
        convert_continuous_to_n_categorical(
            test_model(test_path=validation_csv_fpath)
        )
    )

    result_f1_score = evalution(validation_predicate, validation_original, label_columns)

    return result_f1_score


if __name__ == '__main__':
    result_f1 = get_validation_set_f1_score()
    print('validation result is f1-score: {}'.format(result_f1))
