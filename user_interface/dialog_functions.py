from parse.nltk_parser import corenlp_parse


def update_parse(text, window):
    pos, ner, dp = corenlp_parse(text)
    window["-INPUT_TEXT-"].update(text)
    window["-POS_PARSED-"].update(pos)
    window["-NER_PARSED-"].update(ner)
    window["-DEP_PARSED-"].update(dp)


def update_next(text, window):
    window["-INPUT_TEXT-"].update(text)
    update_parse(text, window)