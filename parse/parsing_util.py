import requests
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPParser, CoreNLPServer
import util
from data.data_processing import get_samples


def corenlp_server_start(path_to_jar=None,
                         path_to_models_jar=None):
    """start corenlp server"""
    # Only in testing phase
    path_to_jar = 'C:/Users/liuda/Documents/Files/StudiumInDtl/Masterarbeit/corenlp/stanford-corenlp-4.5.0/stanford-corenlp-4.5.0.jar'
    path_to_models_jar = 'C:/Users/liuda/Documents/Files/StudiumInDtl/Masterarbeit/corenlp/stanford-corenlp-4.5.0/stanford-corenlp-4.5.0-models.jar'
    if not path_to_jar:
        path_to_jar = input("input the path to stanford-corenlp-4.5.0.jar/n")
    if not path_to_models_jar:
        path_to_models_jar = input("input the path to stanford-corenlp-4.5.0-model.jar/n")
    # Server
    server = CoreNLPServer(path_to_jar=path_to_jar,
                           path_to_models_jar=path_to_models_jar)
    print("Starting corenlp server ...")
    server.start()
    return server


def corenlp_server_stop(server):
    """stop corenlp server"""
    server.stop()
    print("Corenlp server stopped...")


def server_is_running(url):
    """return True if corenlp server is running, otherwise False"""
    try:
        page = requests.get(url)
        status_code = page.status_code
    except Exception as e:
        print(e)
        return False
    print(f"status code from {url} is {status_code}")
    if status_code == 200:
        print("server running, ready to parse")
        return True
    else:
        print("server not running")
        return False


def phrases_from_dep_graph(words, dep_graph):
    """Construct phrases from parsing result.
    (dependency parsing, POS and NER)"""
    phrases_dict = {}
    for i in range(len(words)):
        phrases_dict[i] = []
        # for loop start from 1 to sentence length
        dep_word = dep_graph.get_by_address(i + 1)
        deps = dep_word['deps']
        # todo: here check the pos of the word and apply some rules
        dep_idxs = [ele - 1 for sublist in deps.values() for ele in sublist]
        for idx in dep_idxs:
            phrase = words[idx] + ' ' + words[i]
            phrases_dict[i].append(phrase)
    return phrases_dict


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
    # print(tagged)
    # raw.pretty_print()
    # interpret_dp(raw.nodes)
    return tagged


def corenlp_parse(text):
    # Start server if not started
    if not server_is_running("http://localhost:9000/"):
        corenlp_server_start()
    # pos parsing
    pos = part_of_speech_parsing(text=text)
    # ner parsing
    ner = name_entity_recognition(text=text)
    # dependency parsing
    dp = dependency_parsing(text=text)
    return pos, ner, dp


def get_labels_ts_phrases(testing_file, num, do_shuffle):
    """
    return num of used texts, labels and parsed_phrases as lists
    testing_file: path to test.josn, determined by dataset and laebls_version,
    some phrase results can not be used, sample new ones util the num is reached
    """
    utexts, ulabels, parsed_phrases = [], [], []
    while len(utexts) < num:
        parse_by_num(testing_file=testing_file,
                     num=num,
                     utexts=utexts, ulabels=ulabels,
                     parsed_phrases=parsed_phrases,
                     do_shuffle=do_shuffle)
    return utexts, ulabels, parsed_phrases


def parse_by_num(testing_file, num, utexts, ulabels, parsed_phrases, do_shuffle):
    """add legal result to utexts, ulabels and parsed_phrases"""
    if len(utexts) != 0:
        # if already sampled but num not satisfied, randomly sample square num of the missing samples
        num = (num - len(utexts)) ** 2
        do_shuffle = True
    texts, labels = get_samples(file_path=testing_file,
                                model_name="parsing",
                                num=num,
                                do_shuffle=do_shuffle)
    dep_result, _, ner_result = parse_samples(texts)
    for ith_exp in range(len(texts)):
        num_evaluated = len(utexts)
        if num == num_evaluated:
            # if enough number of parsed samples, return, otherwise sample
            return utexts, ulabels, parsed_phrases
        assert len(texts[ith_exp].split(' ')) == len(labels[ith_exp])
        dp_graph = dep_result[ith_exp]
        ner_labels = ner_result[ith_exp]
        # compare dependency graph length with labels length
        min_add = len(labels[ith_exp]) - 3
        address = min_add
        while dp_graph.contains_address(min_add):
            address = min_add
            min_add = min_add + 1
        # if length not match, skip this example
        if address != len(labels[ith_exp]):
            print(f"{ith_exp}th example's length not match")
            continue
        phrases = interpret_dep(dp_graph, ner_labels, with_ner=True)
        parsed_phrases.append(phrases)
        utexts.append(texts[ith_exp])
        ulabels.append(labels[ith_exp])


def parse_samples(text):
    """input is a list of input text, return the parsing result as a list"""
    # load nlp server
    if not server_is_running("http://localhost:9000/"):
        corenlp_server_start()
    dep_result, pos_result, ner_result = [], [], []
    for sentence in text:
        pos, ner, dp = corenlp_parse(sentence)
        dep_result.append(dp)
        pos_result.append(pos)
        ner_result.append(ner)
    return dep_result, pos_result, ner_result


# todo: change labels to simplified labels?
def ground_truth(labels, dataset, labels_version):
    """return a list of labels as ground truth"""
    labels_path = "data/" + dataset + "/labels/labels" + \
                  labels_version + ".csv"
    f = open(labels_path, 'r')
    import csv
    data = csv.reader(f)
    # generate labels-gt match with dict
    label2gt = {}
    for line in data:
        label2gt[line[0]] = line[1]
    gt = []
    for label in labels:
        try:
            gt.append(label2gt[label])
        except KeyError:
            gt.append('O')
    f.close()
    return gt


def interpret_dep(dependency_graph, ner_labels, with_ner=False):
    """return a list of phrases constructed from dependency graph"""
    phrases = []
    address = 1
    # if current index has phrase, create a list of the words in the phrase
    # and store under the index key
    matching_dict = {}
    while True:
        word_dict = dependency_graph.get_by_address(address)
        if word_dict['address'] is None:
            break
        phrase = ""
        gt = ""
        # skip this word or not
        if not pos_skip(word_dict):
            # check dependency type, dep is a dep_dict
            idxs = []
            for key, value in word_dict['deps'].items():
                if not dep_skip(key):
                    # if not skip, construct phrase
                    for idx in value:
                        idxs.append(idx)
            idxs.append(address)
            idxs.sort()
            phrase = construct_phrase(dependency_graph, idxs)
        if with_ner and phrase != "":
            if ner_labels[address - 1][1] != "O":
                phrase = phrase + ' ' + ner_labels[address - 1][1]
        phrases.append(phrase)
        address = address + 1
    return phrases


def construct_phrase(dependency_graph, idxs):
    words = []
    for i in idxs:
        words.append(dependency_graph.get_by_address(i)['word'])
    return ' '.join(words)


def pos_skip(word_dict):
    """return True if the word has no dependency or labeled
    as VPB, otherwise False"""
    if len(word_dict["deps"]) == 0:
        return True
    if word_dict["ctag"] == "VBP" or word_dict["ctag"] == "VB":
        return True
    # if word_dict["ctag"] == "VB" -> goal action
    return False


def dep_skip(dep):
    skip_list = ["nsubj", "conj", "mark", "obl", "cc", "nmod", "iobj", "cop", "det"]
    if dep in skip_list:
        return True
    return False
