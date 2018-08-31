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

model = load_model(C.MODEL_NAME)

tokenizer = TokenizerSaver.load(C.TOKENIZER_NAME)
tokenizer.oov_token = None

test_path = C.TEST_PATH
test = pd.read_csv(test_path)

X_test = test[C.X].fillna("fillna").values

X_test = tokenizer.texts_to_sequences(X_test)
x_test = sequence.pad_sequences(X_test, maxlen=C.MAX_LEN)
y_pred = model.predict(x_test, batch_size=1024)
test[C.Y] = y_pred

result_dir = 'result'
test.to_csv(os.path.join(result_dir, 'result_{}.csv'.format(time.time())), index=False)
print('predicate finished!')
