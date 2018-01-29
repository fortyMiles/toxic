import pandas as pd
import numpy as np
from keras.models import Model, Input
from keras.layers import Dense, SpatialDropout1D, Dropout
from keras.layers import Embedding, GlobalMaxPool1D, BatchNormalization
import os
import argparse
from fasttext_bn.initial_train_test_data import get_train_test

os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6'


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent neural network for identifying and classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")

    args = parser.parse_args()

    max_features = 50000
    maxlen = 150
    batch_size = 32
    embedding_dim = 64
    epochs = 10

    X_train, y_train, X_test = get_train_test(args.train_file_path, args.test_file_path, max_features, maxlen)

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

    val_acc = np.mean(hist.history['val_acc'])
    submission.to_csv('submission_bn_fasttext_{}.csv'.format(val_acc), index=False)


if __name__ == '__main__':
    main()
