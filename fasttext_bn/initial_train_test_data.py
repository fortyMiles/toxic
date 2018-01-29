import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tools.pickle_dataset import load_from_pickle


@load_from_pickle
def get_train_test(train_csv, test_csv, max_features, maxlen):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    __NAN__ = '__NAN__'
    X_train = train_df['comment_text'].fillna(__NAN__).values
    y_train = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    X_test = test_df['comment_text'].fillna(__NAN__).values

    tok = Tokenizer(num_words=max_features)
    tok.fit_on_texts(list(X_train) + list(X_test))

    x_train = tok.texts_to_sequences(X_train)
    x_test = tok.texts_to_sequences(X_test)

    print('average sequence length is {} '.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('average sequence length is {}'.format(np.mean(list(map(len, x_test)))))

    X_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    return X_train, y_train, X_test


