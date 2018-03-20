import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.preprocessing import sequence
from tools.pickle_tools import TokenizerSaver
import jieba
from keras.models import load_model
import numpy as np

print('keras backend ')
jieba.load_userdict('dic.txt')

model = load_model('model/bank_classification.h5')

x = np.array([[0, 0, 0, 0, 0, 0, 0, 622, 27, 153]])
model.predict(x)


tokenizer = TokenizerSaver.load()
tokenizer.oov_token = None


def get_string_tokenizer(string):
    string = ' '.join(map(lambda x: x.strip(), jieba.cut(string)))
    maxlen = 10
    global tokenizer
    X_test = tokenizer.texts_to_sequences([string])
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    return x_test


def get_classify_result(x):
    global model
    res = model.predict(x)[0]
    return res[0] >= res[1]


def is_bank_related(string):
    assert isinstance(string, str)
    return get_classify_result(get_string_tokenizer(string))


if __name__ == '__main__':
    while True:
        input_string = input('请输入句子(输入q退出)')
        if input_string.upper() == 'Q': break

# input_string = '测试测试'

        result = is_bank_related(input_string)

        print('预测结果： {} 银行相关问题'.format(result))




