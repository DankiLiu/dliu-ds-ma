from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPParser
from parse import util

def part_of_speech_parsing(text: str):
    pos_parser = CoreNLPParser(url='http://localhost:9000',
                               tagtype='pos')
    tokens = pos_parser.tokenize(text)
    tagged = pos_parser.tag(tokens)
    print(f"pos tag\n{text}\n{tagged}")
    return tagged


def name_entity_recognition(text: str):
    ner_parser = CoreNLPParser(url='http://localhost:9000',
                               tagtype='ner')
    tokens = ner_parser.tokenize(text)
    tagged = ner_parser.tag(tokens)
    print(f"ner tag\n{text}\n{tagged}")
    return tagged


def dependency_parsing(text: str):
    # Parse
    parser = CoreNLPDependencyParser()
    tagged = next(parser.parse_text(text))
    tree = tagged.tree()
    tree.pprint()
    print(tagged)
    # raw.pretty_print()
    # interpret_dp(raw.nodes)
    return tagged


def corenlp_parse(text):
    # Start server if not started
    if not util.server_is_running("http://localhost:9000/"):
        util.corenlp_server_start()
    # pos parsing
    pos = part_of_speech_parsing(text=text)
    # ner parsing
    ner = name_entity_recognition(text=text)
    # dependency parsing
    dp = dependency_parsing(text=text)
    return pos, ner, dp


def interpret_dp(raw):
    """
    Interpret dependency parsing
    return action and objects
        intent: the intent action,
        indirect: to or for whom or what the action is performed,
        objects: objects of the action
    """
    interpretation = []
    for word in raw.values():
        # find verb with object(s) as dependency(ies)
        if "VB" in word['ctag']:
            print(word["word"], " - ", word["ctag"])
            if len(word['deps']['obj']) != 0:
                intention = {"action": word['word'], "obj": [], "obl": []}
                # print objects
                for index in word['deps']['obj']:
                    intention["obj"].append(raw[index]['word'])
                if len(word['deps']['obl']) != 0:
                    # print object relations
                    for index in word['deps']['obl']:
                        intention["obl"].append(raw[index]['word'])
                interpretation.append(intention)
    print(interpretation)
