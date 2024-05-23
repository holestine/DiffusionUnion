from tkinter import *
from idlelib.tooltip import Hovertip
import numpy as np

def create_number_control(parent, default, text, tip="", increment=1, type=int, min=-np.Infinity, max=np.Infinity):

    def decrease():
        new_value = type(round((type(entry.get())-type(increment))*100)/100)
        if new_value >= min:
            value.set(new_value)

    def increase():
        new_value = type(round((type(entry.get())+type(increment))*100)/100)
        if new_value <= max:
            value.set(new_value)

    def filter_decimals_only(value):
        if value == "":
            return True
        
        try:
            value = float(value)
            if value >= min and value <= max:
                return True
        except:
            return False
        
        return False

    def filter_ints_only(value):
        if value == "":
            return True
        
        if str.isdigit(value):
            value = int(value)
            if value >= min and value <= max:
                return True
        
        return False

    def verify_valid_value(event):
        if event.widget.get() == '':
            value.set(default)

    # Create frame for the tooltip and other controls
    frame = Frame(parent, bg='grey')
    Hovertip(frame, tip)

    # Add the descriptive label
    Label(frame, text=text, anchor=W).pack(side=LEFT, fill=Y, expand=False)

    # Create a button to decrease the value
    Button(frame, text="-", command=decrease).pack(side=LEFT, fill=BOTH, expand=True)

    # Create the variable and associate it with an Entry control
    value = StringVar(frame, default)
    (width, event) = (4, filter_ints_only) if type==int else (6, filter_decimals_only)
    entry = Entry(frame, width=width, validate='all', validatecommand=(parent.register(event), '%P'), textvariable=value)
    entry.pack(side=LEFT, fill=BOTH, expand=True)
    entry.bind("<FocusOut>", verify_valid_value)
        
    # Create a button to increase the value
    Button(frame, text="+", command=increase).pack(side=LEFT, fill=BOTH, expand=True)
    frame.pack(side=LEFT, fill=BOTH, expand=False)
    return entry

def create_toolbar_button(parent, text, event_handler, tooltip):
    button = Button(parent, text=text, command=event_handler)
    Hovertip(button, tooltip)
    button.pack(side=LEFT, fill=X, expand=False)
    return button
