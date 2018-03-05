import numpy as np

np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, CuDNNGRU
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from toxic.train_utils import train_folds
from tools.initial_train_test_data import get_train_test_and_embedding

import warnings

warnings.filterwarnings('ignore')

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['OMP_NUM_THREADS'] = '4'

# EMBEDDING_FILE = '/data/yuchen/w2v/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
EMBEDDING_FILE = '/data/yuchen/bank/bank_w2v_model.vec'
train_path, test_path = '/data/yuchen/bank/train.csv', '/data/yuchen/bank/test.csv'
submission = '/data/yuchen/bank/sample_submission.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(submission)

X_train = train["comment_text"].fillna("fillna").values

# predicate_fields =["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
predicate_fields =["trans", "not_trans"]

y_train = train[predicate_fields].values
X_test = test["comment_text"].fillna("fillna").values

max_features = 30000
maxlen = 10
embed_size = 200

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
print(len(word_index))
print(embedding_matrix.shape)
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))


# class EarlyStopWithRocAuc(Callback):
#     def __init__(self, moniter='val_loss', ):


def get_model():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()

batch_size = 32
epochs = 15

# X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
# RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

models, scores = train_folds(x_train, y_train, epochs, fold_count=1, batch_size=batch_size, get_model_func=get_model)
# hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
#                  callbacks=[RocAuc], verbose=2)

model = models[0]
y_pred = model.predict(x_test, batch_size=1024)
submission[predicate_fields] = y_pred
submission.to_csv('bank_submission.csv', index=False)