import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tools.pickle_dataset import load_from_pickle
from keras import Input
from keras.layers import Embedding
from toxic.embedding_utils import read_embedding_list, clear_embedding_list, convert_tokens_to_ids
from toxic.nltk_utils import tokenize_sentences


def get_train_test_and_embedding(train_csv, test_csv, sequence_length, vocab_size=None, embedding_file=None, embedding_dim=None):
    assert (embedding_file or embedding_dim) is not None

    if embedding_file is None and embedding_dim is not None:
        X_train, y_train, X_test = _get_train_test(train_csv, test_csv, vocab_size, sequence_length)
        return X_train, y_train, X_test, None
    else:
        X_train, y_train, X_test, embedding = _get_train_test_and_embedding(train_csv, test_csv, sequence_length, embedding_file)
        return X_train, y_train, X_test, embedding


@load_from_pickle
def _get_train_test(train_csv, test_csv, max_features, maxlen):
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


@load_from_pickle
def _get_train_test_and_embedding(train_csv, test_csv, sentence_length, embedding_file=None):
    UNKNOWN_WORD = "_UNK_"
    END_WORD = "_END_"
    NAN_WORD = "_NAN_"

    CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
    list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
    y_train = train_data[CLASSES].values

    print("Tokenizing sentences in train set...")
    tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})

    print("Tokenizing sentences in test set...")
    tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

    words_dict[UNKNOWN_WORD] = len(words_dict)

    print("Loading embeddings...")
    embedding_list, embedding_word_dict = read_embedding_list(embedding_file)
    embedding_size = len(embedding_list[0])
    print("Preparing data...")
    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)

    embedding_matrix = np.array(embedding_list)

    id_to_word = dict((id, word) for word, id in words_dict.items())
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        sentence_length)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        sentence_length)
    X_train = np.array(train_list_of_token_ids)
    X_test = np.array(test_list_of_token_ids)

    return X_train, y_train, X_test, embedding_matrix
