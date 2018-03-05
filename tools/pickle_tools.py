import pickle


class TokenizerSaver:

    token_name = 'model/tokenizer.pickle'

    @staticmethod
    def save(tokenizer):
        with open(TokenizerSaver.token_name, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load():
        with open(TokenizerSaver.token_name, 'rb') as handle:
            tokenizer = pickle.load(handle)
            return tokenizer