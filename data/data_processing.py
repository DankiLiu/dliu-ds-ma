

def jointslu_per_line(line: str):
    """
    process data from JointSLU dataset
    :param line: one line of data read from .iob file
    :return: sentence: a string without EOS and BOS
            words: a list of words without EOS and BOS
            labels: a list of labels, BIO tags
    words and labels should have same length (one to one match)
    """
    # Split the sentence and labels by tab
    sentence, labels = line.split('\t')

    # Strip BOS, EOS labels, first and last labels
    words_list, labels_list = sentence.split(' '), labels.split(' ')
    words_len, labels_len = len(words_list), len(labels_list)
    assert words_len == labels_len

    words = words_list[1: words_len-1]
    labels = labels_list[1: labels_len-1]
    sentence = " ".join(words)
    return sentence, words, labels
