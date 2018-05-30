from jieba import posseg


def is_important(tag):
    not_begin = ['e', 'u', 'r', 'w', 'p', 'x']
    return not any([tag.startswith(b) for b in not_begin])


def filter_unimportant(string):
    words = [
        x[0] for x in [tuple(t) for t in posseg.cut(string)]
        if is_important(x[1])
    ]
    # print(words)
    string = ' '.join(map(lambda x: x.strip(), words))
    return string


if __name__ == '__main__':
    print(filter_unimportant('什么是信用卡?'))
    print(filter_unimportant('它的附卡是什么？'))
