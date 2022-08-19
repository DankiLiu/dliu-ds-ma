import PySimpleGUI as sg


def parsing_layout():
    """
        pos-tag layout:
                    |   Input   ******
            POS-TAG |   Parsed  ******
                    |   Action  ******
                    |   Objects ** **
    """
    parsing_left = [sg.Text("Parsing",
                            font=("Arial bold", 15),
                            text_color="black")]
    parsing_tags = [sg.Column([[sg.Text(" InputText ",
                                        font=("arial", 11),
                                        text_color="blue",
                                        background_color="lightgrey")],
                               [sg.Text(" POS-tag ",
                                        font=("arial", 11),
                                        text_color="blue",
                                        background_color="lightgrey")],
                               [sg.Text(" NER-tag ",
                                        font=("arial", 11),
                                        text_color="blue",
                                        background_color="lightgrey")],
                               [sg.Text(" DEP-tag ",
                                        font=("arial", 11),
                                        text_color="blue",
                                        background_color="lightgrey")]
                               ]),
                    sg.Column([[sg.Text("Input text here ...",
                                        size=7,
                                        auto_size_text=True,
                                        key="-INPUT_TEXT-")],
                               [sg.Text(" POS tags here ...",
                                        size=7,
                                        auto_size_text=True,
                                        key="-POS_PARSED-")],
                               [sg.Text(" NER tags here ...",
                                        size=7,
                                        auto_size_text=True,
                                        key="-NER_PARSED-")],
                               [sg.Multiline(key="-DEP_PARSED-",
                                             auto_size_text=True)]])
                    ]
    parsing_results = [sg.Column([[sg.Text("Action",
                                           font=("Arial", 11),
                                           text_color="blue",
                                           background_color="lightgrey")],
                                  [sg.Text("Objects",
                                           font=("Arial", 11),
                                           text_color="blue",
                                           background_color="lightgrey"
                                           )]]),
                       sg.Column([[sg.InputText("Output action from model ...", key="-POS_ACTION-")],
                                  [sg.InputText("Object name1, name2, ...", key="-POS_OBJECTS-")]])]
    PARSING_LAYOUT = [
        parsing_left,
        parsing_tags,
        parsing_results
    ]
    return PARSING_LAYOUT
