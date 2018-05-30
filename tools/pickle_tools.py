import pickle


class TokenizerSaver:

    token_name = 'model/tokenizer.pickle'

    @staticmethod
    def save(tokenizer, token_name=None):
        token_name = token_name or TokenizerSaver.token_name
        with open(token_name, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(token_name=None):
        token_name = token_name or TokenizerSaver.token_name
        with open(token_name, 'rb') as handle:
            tokenizer = pickle.load(handle)
            return tokenizer