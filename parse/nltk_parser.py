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
    result = next(parser.parse_text(text))
    print(result)
