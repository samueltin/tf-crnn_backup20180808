import os

DICT_FILE =os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dict', 'full_dict.txt')

def dict_as_str():
    corpus_file = open(DICT_FILE, mode='r', encoding='utf-8')
    content = corpus_file.read()
    corpus_file.close()
    list = content.split('\n')
    return ''.join(list)
