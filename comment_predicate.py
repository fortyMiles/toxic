# Created by mqgao at 2018/8/31

"""
Feature: #Enter feature name here
# Enter feature description here

Scenario: #Enter scenario name here
# Enter steps here

Test File Location: # Enter

"""
from keras.preprocessing import sequence
from tools.pickle_tools import TokenizerSaver
from keras.models import load_model
import pandas as pd
import config as C
import os
import time
from tools.convert_category import convert_continuos_to_2_categorical


def test_model(model=None, model_path=None, token_name=None, test_path=None, max_len=None, save=True):
    if model is None:
        model_path = model_path or C.MODEL_NAME
        token_name = token_name or C.TOKENIZER_NAME
        test_path = test_path or C.TEST_PATH
        max_len = max_len or C.MAX_LEN

        model = load_model(model_path)
    else:
        model = model  # model assigned to argument model

    tokenizer = TokenizerSaver.load(token_name)
    tokenizer.oov_token = None

    test = pd.read_csv(test_path)

    X_test = test[C.X].fillna("fillna").values

    X_test = tokenizer.texts_to_sequences(X_test)
    x_test = sequence.pad_sequences(X_test, maxlen=max_len)
    y_pred = model.predict(x_test, batch_size=1024)
    test[C.Y] = y_pred

    test = convert_continuos_to_2_categorical(test)
    if save:
        result_dir = 'result'
        test.to_csv(os.path.join(result_dir, 'result_{}.csv'.format(time.time())), index=False)

    print('predicate finished!')

    return test


if __name__ == '__main__':
    test_model()

