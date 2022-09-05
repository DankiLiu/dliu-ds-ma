import parse.util as util
from data.data_processing import jointslu_per_line, \
    read_jointslu_lines, find_all_labels
import parse.nltk_parser as np


def corenlp_parse(text):
    # pos parsing
    np.part_of_speech_parsing(text=text)
    # ner parsing
    np.name_entity_recognition(text=text)
    # dependency parsing
    np.dependency_parsing(text=text)


def parse():
    # Start server if not started
    if not util.server_is_running("http://localhost:9000/"):
        util.corenlp_server_start()
    dataset_path = "../data/sample.iob"
    lines = read_jointslu_lines(dataset_path)

    for line in lines:
        sentence, words, labels = jointslu_per_line(line)
        print(f"sentence {sentence}")
        print(f"words    {words}")
        print(f"labels   {labels}")
        corenlp_parse(sentence)
        if_continue = input("press enter to continue, 'n' to stop")
        if if_continue == "":
            continue
        else:
            break


def main():
    # read the file
    dataset_path = "../data/atis.train.w-intent.iob"
    lines = read_jointslu_lines(dataset_path)
    # get the labels
    labels_l = []
    for line in lines:
        _, _, labels = jointslu_per_line(line)
        labels_l.append(labels)
    # save the labels into file
    find_all_labels(labels_l)


if __name__ == '__main__':
    parse()
