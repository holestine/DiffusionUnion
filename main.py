from tkinter import *                   
from tkinter import ttk
import torch
import os

import image_inpainting as inpainting
import image_generation as generation
import image_depth      as depth
import image_to_vid

# Disable warning, may be bug in some versions of PyTorch
torch.backends.cudnn.enabled = False

DEV_MODE = False

GENERATION_TAB_NAME  = 'Image Generation'
INPAINTING_TAB_NAME  = 'Inpainting'
DEPTH_TAB_NAME       = 'Depth'
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

        # Create a tabbed interface
        self.tabControl = ttk.Notebook(root)

        generation_tab = Frame(self.tabControl)
        self.generation_tab = generation.image_generation_ui(generation_tab, self.history)
        self.tabControl.add(generation_tab, text=GENERATION_TAB_NAME) 

        inpainting_tab = Frame(self.tabControl)
        self.inpainting_tab = inpainting.inpainting_ui(inpainting_tab, self.history)
        self.tabControl.add(inpainting_tab, text=INPAINTING_TAB_NAME) 

        depth_tab = Frame(self.tabControl)
        self.depth_tab = depth.image_depth_ui(depth_tab, self.history)
        self.tabControl.add(depth_tab, text=DEPTH_TAB_NAME)
        
        if DEV_MODE:
            image_to_vid_tab = Frame(self.tabControl)
            self.image_to_vid_tab = image_to_vid.image_to_vid(image_to_vid_tab)
            self.tabControl.add(image_to_vid_tab, text=IMAGE2VIDEO_TAB_NAME) 
            
        self.tabControl.pack(expand=True, fill=BOTH) 

        # Used perform any synchronizing activities when a new tab is selected
        root.bind("<<NotebookTabChanged>>", self.on_tab_selected)

    def on_tab_selected(self, event):
        tab_name = event.widget.tab('current')['text']
        if tab_name == GENERATION_TAB_NAME:
            #print('{} selected'.format(tab_name))
            self.generation_tab.refresh_ui()
         
        elif tab_name == INPAINTING_TAB_NAME:
            #print('{} selected'.format(tab_name))
            self.inpainting_tab.refresh_ui()
        
        elif tab_name == DEPTH_TAB_NAME:
            #print('{} selected'.format(tab_name))
            self.depth_tab.refresh_ui()
         
        elif tab_name == IMAGE2VIDEO_TAB_NAME:
            print('{} selected'.format(tab_name))

if __name__ == "__main__":
    # Create Tk UI
    root = Tk()
    app = DiffusionUnionUI(root)
    root.mainloop()

