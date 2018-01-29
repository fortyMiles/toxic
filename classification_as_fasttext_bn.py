import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model, Input
from keras.layers import Dense, SpatialDropout1D, Dropout
from keras.layers import Embedding, GlobalMaxPool1D, BatchNormalization
from keras.preprocessing.text import Tokenizer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

__NAN__ = '__NAN__'

X_train = train_df['comment_text'].fillna(__NAN__).values
y_train = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test_df['comment_text'].fillna(__NAN__).values

max_features = 50000
maxlen = 150
batch_size = 32
embedding_dim = 64
epochs = 4

tok = Tokenizer(num_words=max_features)
tok.fit_on_texts(list(X_train) + list(X_test))

x_train = tok.texts_to_sequences(X_train)
x_test = tok.texts_to_sequences(X_test)

print('average sequence length is {} '.format(np.mean(list(map(len, x_train)), dtype=int)))
print('average sequence length is {}'.format(np.mean(list(map(len, x_test)))))

X_train = sequence.pad_sequences(x_train, maxlen=maxlen)
X_test = sequence.pad_sequences(x_test, maxlen=maxlen)


comment_input = Input(shape=(maxlen, ))
comment_embedding = Embedding(max_features, embedding_dim, input_length=maxlen)(comment_input)
comment_embedding = SpatialDropout1D(0.25)(comment_embedding)
max_emb = GlobalMaxPool1D()(comment_embedding)
main = BatchNormalization()(max_emb)
main = Dense(64)(main)
main = Dropout(0.5)(main)
output = Dense(6, activation='sigmoid')(main)
model = Model(inputs=comment_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
y_pred = model.predict(X_test)


submission = pd.read_csv('data/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred / 1.4

submission.head()
submission.to_csv('submission_bn_fasttext.csv', index=False)

