import PySimpleGUI as sg


def pos_layout():
    """
        pos-tag layout:
                    |   Input   ******
            POS-TAG |   Parsed  ******
                    |   Action  ******
                    |   Objects ** **
    """
    pos_left = [sg.Text("Part-of-Speech-Tagging",
                        font=("Arial bold", 15),
                        text_color="black",
                        background_color="white")]
    pos_tags = [sg.Column([[sg.Text(" Input  ",
                                    font=("arial bold", 11),
                                    text_color="blue",
                                    background_color="lightgrey")],
                           [sg.Text(" Parsed ",
                                    font=("arial bold", 11),
                                    text_color="green",
                                    background_color="lightgrey")]]),
                sg.Column([[sg.Text("Input text here ...",
                                    size=7,
                                    auto_size_text=False,
                                    key="-INPUT_TEXT-")],
                           [
                               sg.Text("Parsed text here ...",
                                       size=7,
                                       auto_size_text=False,
                                       key="-POS_PARSED-")
                           ]])
                ]
    pos_results = [sg.Column([[sg.Text("Action",
                                       font=("Arial bold", 11),
                                       text_color="blue",
                                       background_color="lightgrey")],
                              [sg.Text("Objects",
                                       font=("Arial bold", 11),
                                       text_color="blue",
                                       background_color="lightgrey"
                                       )]]),
                   sg.Column([[sg.InputText("Output action from model ...", key="-POS_ACTION-")],
                              [sg.InputText("Object name1, name2, ...", key="-POS_OBJECTS-")]])]
    POS_LAYOUT = [
        pos_left,
        pos_tags,
        pos_results
    ]
    return POS_LAYOUT
