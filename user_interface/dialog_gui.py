import PySimpleGUI as sg
import user_interface.layouts as ly
# Define the layout of command input area
command_input = [[sg.Text("This is a GUI the of robot task-oriented dialog system.")],
                 [sg.Text("Please input your command below",
                          text_color="blue",
                          font=("arial bold", 11),
                          background_color="lightgrey")],
                 [sg.InputText(default_text="Put that milk into the fridge.", key="-INPUT_TEXT-"),
                  sg.Button("OK")]]

# Define the layout of the model output
# model_output shows the original sentence and the parsed sentence.
model_layout = ly.pos_layout()
layout = [[sg.Column(command_input)],
          [sg.HSeparator()],
          [sg.Column(model_layout)]]
# Create the window
window = sg.Window(title="Dialog GUI", layout=layout)

# Create an event loop
while True:
    event, values = window.read()
    if event == "OK" or event == sg.WIN_CLOSED:
        break

window.close()
