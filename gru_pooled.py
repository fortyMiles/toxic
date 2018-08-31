import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, CuDNNGRU
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from toxic.train_utils import train_folds
from tools.pickle_tools import TokenizerSaver
import tensorflow as tf
from tools import string_tools
import warnings
import os
import config as C
import string
import logging

np.random.seed(42)
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# os.environ['OMP_NUM_THREADS'] = '4'

# EMBEDDING_FILE = '/data/yuchen/w2v/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

TRAIN_CORPUS = 'train_corpus'
MODEL = 'model'

EMBEDDING_FILE = C.EMBEDDING_FILE
TRAIN_PATH = C.TRAIN_PATH
# ROOT_PATH = '/Users/kouminquan/Workspaces/IBM/dataset'
# test_path = '{}/bank_test.csv'.format(root_path)
# submission = '{}/submission.csv'.format(root_path)

PREDICATE_FIELDS = C.Y

train = pd.read_csv(TRAIN_PATH)
# test = pd.read_csv(test_path)
# submission = pd.read_csv(submission)

logging.info('read csv finished!')

X_train = train[C.X].fillna(string.whitespace).values
# X_train = [string_tools.filter_unimportant(x) for x in X_train]

# predicate_fields =["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train[PREDICATE_FIELDS].values

max_features = 200000

# X_test = test["comment_text"].fillna("fillna").values
logging.info('begin tokenizer')

tokenizer = text.Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(list(X_train) + list(X_test))
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=C.MAX_LEN)
# x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

TokenizerSaver.save(tokenizer, C.TOKENIZER_NAME)
print('load tokenizer finish')


def get_coefs(word, *arr): return word, np.asarray(arr, dtype=np.float32)


logging.info('begin read word2vec')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, C.EMBED_SIZE))
print(len(word_index))
print(embedding_matrix.shape)
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        try:
            embedding_matrix[i] = embedding_vector
        except IndexError:
            pass

logging.info('end read word2vec')


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
    inp = Input(shape=(C.MAX_LEN,))
    x = Embedding(max_features, C.EMBED_SIZE, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(320, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(len(PREDICATE_FIELDS), activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()
batch_size = 32
epochs = 20
early_stop = 5

# X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
# RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

models, scores = train_folds(x_train, y_train, epochs,
                             fold_count=1, batch_size=batch_size,
                             get_model_func=get_model, evaluation='auc', early_stop=early_stop)
# hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
#                  callbacks=[RocAuc], verbose=2)

model = models[0]
graph = tf.get_default_graph()
model.save(C.MODEL_NAME)


# y_pred = model.predict(x_test, batch_size=1024)
# submission[predicate_fields] = y_pred
# submission.to_csv('bank_submission.csv', index=False)

