from keras import backend as K
import os
from importlib import reload


# def set_keras_backend(backend):
#     if K.backend() != backend:
#         os.environ['KERAS_BACKEND'] = backend
#         reload(K)
#         assert K.backend() == backend
#
#
# set_keras_backend("theano")


from keras.preprocessing import sequence
from tools.pickle_tools import TokenizerSaver
import jieba
from keras.models import load_model
import numpy  as np
import conf.logger_config as log_conf
logger = log_conf.get_logger_root()
import random
from tools.string_tools import filter_unimportant

logger.info('keras backend is {}'.format(K.backend()))

path = 'model'
jieba.load_userdict('{}/dic.txt'.format(path))

model = load_model('{}/bank_classification.h5'.format(path))

x = np.array([[0, 0, 0, 0, 0, 0, 0, 622, 27, 153, 1, 1, 1, 1, 1]])
model.predict(x)


tokenizer = TokenizerSaver.load()
tokenizer.oov_token = None

DENSITY, VALUE = 'density', 'value'


def get_string_tokenizer(string):
    string = filter_unimportant(string)
    maxlen = 15
    global tokenizer
    X_test = tokenizer.texts_to_sequences([string])
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    return x_test


def get_type(string):
    return model.predict(get_string_tokenizer(string))[0]


def is_bank(x, metric=DENSITY):
    global model
    res = model.predict(x)[0]

    density = res[0]/sum(1 for i in x[0] if i != 0)
    density_threshold = 0.1
    true_threshold = 0.3
    #
    # if (res[0] / res[1]) >= true_prop_threshold
    logger.info('res == {}'.format(res))
    logger.info('density == {}'.format(density))

    # return res[0] >= true_threshold or density >= density_threshold
    return res[0] > true_threshold or density >= density_threshold
    # if metric == VALUE:
    #     return res[0] / res[1] >= true_prop_threshold
    # elif metric == DENSITY:
    #     logger.info('DENSITY: {}'.format(density))
    #     return density >= density_threshold
    # else:
    #     raise TypeError('error metric type supported <density, value>')


def is_dxh(string):
    try:
        return np.argmax(get_type(string)) == 2
    except Exception as e:
        print(e)
        return False


def is_daily(string, metric=DENSITY):
    assert isinstance(string, str)
    logger.info('STRING CLASSIFIER: {}'.format(string))
    try:
        result = is_bank(get_string_tokenizer(string), metric=metric)
        logger.info('STRING {} IS_BANK {}'.format(string, result))
        return 1 - int(result)
    except Exception as e:
        logger.error(e)
        return random.choice([True, False])


if __name__ == '__main__':
    while True:
        input_string = input('请输入句子(输入q退出)\n:')
        if input_string.upper() == 'Q': break

        result = get_type(input_string)
        print('type is {}'.format(result))
        print('dxh is {}'.format(is_dxh(input_string)))
        #
        # print('预测结果： {} 银行相关问题'.format(1 - result))
