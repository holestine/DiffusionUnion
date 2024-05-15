from tkinter import *
from idlelib.tooltip import Hovertip
import numpy as np

def filter_numbers_only(value):
    try:
        if value == "":
            return True
        else:
            float(value)
            return True
    except:
        return False

def filter_ints_only(value):
    if str.isdigit(value) or value == "":
        return True
    return False
    
def filter_positive_ints_only(value):
    if value == "":
        return True
    if str.isdigit(value):
        if int(value) > 0:
            return True
    return False
        
def create_number_control(parent, default, text, tip="", increment=1, type=int, positive=False, gt_zero=False, max=np.Infinity):
    # Create frame for the tooltip
    frame = Frame(parent, bg='grey')
    Hovertip(frame, tip)

    # Add the label
    Label(frame, text=text, anchor=W).pack(side=LEFT, fill=Y, expand=False)

    def decrease():
        new_value = round((type(entry.get())-type(increment))*100)/100

        if not positive:
            x.set(new_value)
        elif positive:
            if gt_zero:
                if new_value > 0:
                    x.set(new_value)
            elif new_value >= 0:
                x.set(new_value)

    def increase():
        new_value = round((type(entry.get())+type(increment))*100)/100
        if new_value <= max:
            x.set(new_value)

    # Create a button to decrease the value
    Button(frame, text="-", command=decrease).pack(side=LEFT, fill=BOTH, expand=True)

    # Create the variable and associate it with the control
    if type == int:
        x = IntVar(frame, default)
        entry = Entry(frame, width=4, validate='all', validatecommand=(parent.register(filter_positive_ints_only), '%P'), textvariable=x)
    else:
        x = StringVar(frame, default)
        entry = Entry(frame, width=6, validate='all', validatecommand=(parent.register(filter_numbers_only), '%P'), textvariable=x)
        
    # Align everything to the left and fill the space
    entry.pack(side=LEFT, fill=BOTH, expand=True)
    Button(frame, text="+", command=increase).pack(side=LEFT, fill=BOTH, expand=True)
    frame.pack(side=LEFT, fill=BOTH, expand=False)
    return entry