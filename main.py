from tkinter import *                   
from tkinter import ttk
from PIL import Image
import torch
import time

import numpy as np
import image_inpainting as inpainting
import image_generation as generation
import image_to_vid

# Disable warning, may be bug in some versions of PyTorch
torch.backends.cudnn.enabled = False

DEV_MODE = False

class DiffusionUnionUI:

    def __init__(self, root, background=None):
        self.root = root
        
        # Window title
        self.root.title("Diffusion Union")

        tabControl = ttk.Notebook(root)

        if DEV_MODE:
            generation_tab = Frame(tabControl)
            generation.image_generation_ui(generation_tab)
            tabControl.add(generation_tab, text ='Image Generation') 

        inpainting_tab = Frame(tabControl)
        inpainting.inpainting_tab(inpainting_tab)
        tabControl.add(inpainting_tab, text ='Inpainting') 
        
        if DEV_MODE:
            image_to_vid_tab = Frame(tabControl)
            image_to_vid.image_to_vid(image_to_vid_tab)
            tabControl.add(image_to_vid_tab, text ='Image to Video') 
            
        tabControl.pack(expand=True, fill=BOTH) 


if __name__ == "__main__":
    root = Tk()
    app = DiffusionUnionUI(root)
    root.mainloop()

