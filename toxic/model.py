from keras.layers import Dense, Embedding, Input
from keras.layers import Bidirectional, Dropout, CuDNNGRU
from keras.models import Model
from keras.optimizers import RMSprop
from objectives import roc_auc_score
from keras.layers import GRUCell, RNN, LSTMCell, GlobalAvgPool1D, GlobalMaxPool1D, concatenate
from keras.losses import categorical_crossentropy


def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=True)(input_layer)

    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    avg_pool = GlobalAvgPool1D()(x)
    max_pool = GlobalMaxPool1D()(x)
    conc = concatenate([avg_pool, max_pool])
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(x)
    x = Dense(dense_size, activation="relu")(conc)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=categorical_crossentropy,
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])

    return model
