from tkinter import *                   
from tkinter import ttk
from PIL import Image
import torch
import os

import numpy as np
import image_inpainting as inpainting
import image_generation as generation
import image_to_vid

# Disable warning, may be bug in some versions of PyTorch
torch.backends.cudnn.enabled = False

DEV_MODE = True

GENERATION_TAB_NAME = 'Image Generation'
INPAINTING_TAB_NAME = 'Inpainting'
IMAGE2VIDEO_TAB_NAME = 'Image to Video'

class DiffusionUnionUI:

    def __init__(self, root, background=None):
        self.root = root
        
        # Window title
        self.root.title("Diffusion Union")

        # Used to store generated images
        self.history = []
        if not os.path.exists('history'):
            os.makedirs('history')

        self.tabControl = ttk.Notebook(root)

        generation_tab = Frame(self.tabControl)
        self.generation_tab = generation.image_generation_ui(generation_tab, self.history)
        self.tabControl.add(generation_tab, text=GENERATION_TAB_NAME) 

        inpainting_tab = Frame(self.tabControl)
        self.inpainting_tab = inpainting.inpainting_tab(inpainting_tab, self.history)
        self.tabControl.add(inpainting_tab, text=INPAINTING_TAB_NAME) 
        
        if DEV_MODE:
            image_to_vid_tab = Frame(self.tabControl)
            self.image_to_vid_tab = image_to_vid.image_to_vid(image_to_vid_tab)
            self.tabControl.add(image_to_vid_tab, text=IMAGE2VIDEO_TAB_NAME) 
            
        self.tabControl.pack(expand=True, fill=BOTH) 

        # Used to update selected canvas with last generated image
        root.bind("<<NotebookTabChanged>>", self.on_tab_selected)

    def on_tab_selected(self, event):
        tab = event.widget.tab('current')['text']
        if tab == GENERATION_TAB_NAME:
            print('{} selected'.format(tab))
            self.generation_tab.refresh_canvas()
         
        elif tab == INPAINTING_TAB_NAME:
            print('{} selected'.format(tab))
            self.inpainting_tab.refresh_canvas()
         
        elif tab == IMAGE2VIDEO_TAB_NAME:
            print('{} selected'.format(tab))


        #tabName = event.widget.select()
        #tab = event.widget.tab('current')['text']
        #print('{} selected'.format(tab))

        #tabName = self.tabControl.select()
        #if tabName:
        #    widget = self.tabControl.nametowidget(tabName) 


if __name__ == "__main__":
    root = Tk()
    app = DiffusionUnionUI(root)
    root.mainloop()

