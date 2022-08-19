import PySimpleGUI as sg
import user_interface.layouts as ly
from data.data_processing import read_jointslu_lines, \
    jointslu_per_line
from user_interface.dialog_functions import update_parse, \
    update_next

# Define the layout of command input area
command_input = [[sg.Text("This is a robot task-oriented dialog system GUI.")],
                 [sg.Text("Please input your command below",
                          text_color="black",
                          font=("arial", 11))],
                 [sg.InputText(default_text="Put that milk into the fridge.",
                               key="-INPUT_TEXT-")],
                 [sg.Button("Parse", key="-BT_PARSE-"),
                  sg.Button("Next", key="-BT_NEXT-")]]

# Define the layout of the model output
# model_output shows the original sentence and the parsed sentence.
model_layout = ly.parsing_layout()
layout = [[sg.Column(command_input)],
          [sg.HSeparator()],
          [sg.Column(model_layout)]]
# Create the window
window = sg.Window(title="Dialog GUI", layout=layout)

# Load data
jointslu_lines = read_jointslu_lines()
index = 0


def load_sentence(index):
    sentence, words, labels = \
        jointslu_per_line(jointslu_lines[index])
    return sentence, words, labels


s, _, _ = load_sentence(index)
# Create an event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == "-BT_PARSE-":
        update_parse(text=s,
                     window=window)
    if event == "-BT_NEXT-":
        index = index + 1
        sentence, _, _ = load_sentence(index)
        update_next(text=sentence,
                    window=window)

window.close()
