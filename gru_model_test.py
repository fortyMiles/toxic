# Created by mqgao at 2018/8/31

"""
Feature: #Enter feature name here
# Enter feature description here

Scenario: #Enter scenario name here
# Enter steps here

Test File Location: # Enter

"""
from keras import backend as K
from keras.preprocessing import sequence
from tools.pickle_tools import TokenizerSaver
from keras.models import load_model
import pandas as pd
import config as C
import os
import time

path = 'model'

model = load_model(os.path.join(path, C.MODEL_NAME.format(path)))

# feature_size = 50
# x = np.array([[0]*feature_size])
# model.predict(x)

tokenizer = TokenizerSaver.load()
tokenizer.oov_token = None

test_path = None
test = pd.read_csv(test_path)

X_test = test[C.X].fillna("fillna").values

X_test = tokenizer.texts_to_sequences(X_test)
x_test = sequence.pad_sequences(X_test, maxlen=C.MAX_LEN)
y_pred = model.predict(x_test, batch_size=1024)
x_test[C.Y] = y_pred
x_test.to_csv(os.path.join(path, 'result_{}.csv'.format(time.time())), index=False)
print('predicate finished!')
