from keras import Input
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers import GlobalMaxPool1D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras import Model


def get_model(embedding_matrix, sequence_length, vocab_size, embedding_dim, dropout_rate=0.5, dense_size=64):
    comment_input = Input(shape=(sequence_length, ))

    if embedding_matrix is None:
        comment_embedding = Embedding(vocab_size, embedding_dim, input_length=sequence_length)(comment_input)
    else:
        comment_embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=True)(comment_input)

    comment_embedding = SpatialDropout1D(0.25)(comment_embedding)
    max_emb = GlobalMaxPool1D()(comment_embedding)
    main = BatchNormalization()(max_emb)
    main = Dense(dense_size)(main)
    main = Dropout(dropout_rate)(main)
    output = Dense(6, activation='sigmoid')(main)
    model = Model(inputs=comment_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

