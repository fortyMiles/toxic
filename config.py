__root_path = '/Users/kouminquan/Workspaces/IBM/dataset'
EMBEDDING_FILE = "{}/bank_w2v_model.vec".format(__root_path)
TRAIN_PATH = "{}/bank_train.csv".format(__root_path)

Y = ['is_bank', 'is_daily']
X = 'comment_text'
MAX_LEN = 15
EMBED_SIZE = 200

TOKENIZER_NAME = 'model/tokenizer-0530.tk'
MODEL_NAME = 'model/bank_classification_2018_0530.h5'
