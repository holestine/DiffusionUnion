from tkinter import *                   
from tkinter import ttk
from PIL import Image
import torch
import time

import numpy as np
import inpainting as ip
import image_to_vid as i2v

# Disable warning, may be bug in some versions of PyTorch
torch.backends.cudnn.enabled = False

DEBUG = FALSE

class DiffusionUnionUI:

    def __init__(self, root, background=None):
        self.root = root
        
        # Window title
        self.root.title("Diffusion Union")

        tabControl = ttk.Notebook(root) 
        inpainting_tab = Frame(tabControl) 
        image_to_vid = Frame(tabControl) 
        tabControl.add(inpainting_tab, text ='Inpainting') 
        tabControl.add(image_to_vid, text ='Image to Video') 
        tabControl.pack(expand=TRUE, fill=BOTH) 
        
        

        ip.inpainting_ui(inpainting_tab)
        i2v.image_to_vid(image_to_vid)
        


if __name__ == "__main__":
    root = Tk()
    #app = DiffusionUnionUI(root, "background.png")
    app = DiffusionUnionUI(root)
    root.mainloop()

