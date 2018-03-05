from keras.preprocessing import sequence
from tools.pickle_tools import TokenizerSaver
import jieba
from keras.models import load_model

jieba.load_userdict('dic.txt')


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model = load_model('model/bank_classification.h5')


def get_string_tokenizer(string):
    string = ' '.join(map(lambda x: x.strip(), jieba.cut(string)))
    maxlen = 10
    tokenizer = TokenizerSaver.load()
    X_test = tokenizer.texts_to_sequences([string])
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    return x_test


def get_test_result(x):
    return model.predict(x)


while True:
    input_string = input('请输入句子(输入q退出)')
    if input_string.upper() == 'Q': break

    result = get_test_result(get_string_tokenizer(input_string))

    print('预测结果： {} 银行相关问题'.format(result))




