import parse.util as util

from nltk import word_tokenize, pos_tag
from nltk.parse.corenlp import CoreNLPDependencyParser


def pos_tag(text: str):
    sentence = word_tokenize(text)
    tagged = pos_tag(sentence)
    print(type(tagged))
    print(tagged)
    return tagged


def dependency_parsing(text: str):
    # Parse
    parser = CoreNLPDependencyParser()
    result = list(parser.raw_parse(text))
    print(result)


def main():
    text = "can you make a green tea?"
    # start server
    server = util.corenlp_server_start()
    # parse
    dependency_parsing(text=text)
    # stop server
    util.corenlp_server_stop(server)


if __name__ == '__main__':
    main()