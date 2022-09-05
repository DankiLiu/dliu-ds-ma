from parse import util


def read_jointslu_lines(file_path=None):
    if not file_path:
        file_path = "../data/sample.iob"
    with open(file_path) as f:
        lines = f.readlines()
    return lines


def jointslu_per_line(line: str):
    """
    process data from JointSLU dataset
    :param line: one line of data read from .iob file
           path: path
    :return: sentence: a string without EOS and BOS
            words: a list of words without EOS and BOS
            labels: a list of labels, BIO tags
    words and labels should have same length (one to one match)
    """
    # Split the sentence and labels by tab
    sentence, labels = line.split('\t')
    sentence = sentence.strip()
    labels = labels.strip()
    # Strip BOS, EOS labels, first and last labels
    words_list, labels_list = sentence.split(' '), labels.split(' ')
    words_len, labels_len = len(words_list), len(labels_list)
    assert words_len == labels_len

    words = words_list[1: words_len-1]
    labels = labels_list[1: labels_len-1]
    sentence = " ".join(words)
    return sentence, words, labels


def find_all_labels(llist, boi=False):
    import itertools
    labels = list(itertools.chain(*llist))
    labels = list(filter(lambda l: l != 'O', labels))

    if not boi:
        labels = [label.split('-')[1] for label in labels]
    labels = set(labels)
    print(f"{len(labels)} labels: {labels}")
    util.save_jointslu_labels(labels)
