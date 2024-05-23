from tkinter import *
from tkinter import filedialog
from idlelib.tooltip import Hovertip
import threading
from transformers import SamModel, SamProcessor
from transformers import pipeline
import torch
import time
from diffusers.utils import load_image, make_image_grid
import numpy as np
from PIL import Image

DEBUG = False


def show_masks_on_image(raw_image_path, masks):

    # Internal method to convert the mask to a transparent image
    def get_mask_image(mask):
        # Get a random color and add some transparency 
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

        # Height and width are the last two dimensions
        h, w = mask.shape[-2:]

        # Apply the color to the mask
        mask_image = (255*mask).reshape(h, w, 1) * color.reshape(1, 1, -1)

        # Construct the PIL image 
        mask_image = Image.fromarray(mask_image.astype('uint8'), mode='RGBA')

        return mask_image

    # Load the current image
    raw_image = Image.open(raw_image_path).convert("RGBA")

    # Draw semi-transparent masks in different colors
    for mask in masks:
        m = get_mask_image(mask)
        raw_image.paste(m, (0,0), m)

    return raw_image


class image_segmentation_ui:

    def __init__(self, parent, history, width=512, height=512):

        # Efficient attention is not native in old PyTorch versions and is needed to reduce GPU usage
        self.use_efficient_attention = int(torch.__version__.split('.')[0]) < 2

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Size of image to work with
        self.width = width
        self.height = height

        # Used to store generated images
        self.history = history

        # Get frames needed for layout
        toolbar, left_frame, right_frame = self.create_layout(parent)

        # Populate controls
        self.initialize_toolbar(toolbar)
        self.initialize_prompts(right_frame)
        self.initialize_canvas(left_frame)

        self.update_controls()

    def create_layout(self, parent):
        # Create toolbar
        toolbar = Frame(parent, width=2*self.width, height=20, bg='light grey')
        toolbar.pack(side=TOP, fill=X, expand=False)

        # Create left frame
        left_frame = Frame(parent, width=self.width, height=self.height, bg='grey')
        left_frame.pack(side=LEFT, fill=BOTH, expand=False)

        # Create right frame
        right_frame = Frame(parent, width=self.width, height=self.height, bg='grey')
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        return toolbar, left_frame, right_frame

    def initialize_canvas(self, parent):
        # Create canvas
        self.canvas = Canvas(parent, bg="black", width=self.width, height=self.height)
        self.canvas.pack(fill=BOTH, expand=False)

    def initialize_prompts(self, parent):
        # Create text box for entering the prompt
        prompt = ""
        Label(parent, text="Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.prompt = Text(parent, height=1, wrap=WORD)
        self.prompt.insert(END, prompt)
        self.prompt.pack(side=TOP, fill=BOTH, expand=True)

    def initialize_toolbar(self, toolbar):
        
        # Create combo box for selecting a diffusion model
        checkpoint_frame = Frame(toolbar, bg='grey')
        checkpoint_options = ["Segment Anything"]
        self.checkpoint = StringVar(checkpoint_frame, checkpoint_options[0])
        Hovertip(checkpoint_frame, 'Select the model to use')
        Label(checkpoint_frame, text="Model", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        checkpoint_menu = OptionMenu(checkpoint_frame, self.checkpoint, *checkpoint_options)
        checkpoint_menu.config(width=20)
        checkpoint_menu.pack(side=LEFT, fill=X, expand=True)
        checkpoint_frame.pack(side=LEFT, fill=X, expand=False)

        # Create a button to segment the image
        self.segment_button = Button(toolbar, text="Segment", command=self.segment)
        Hovertip(self.segment_button, 'Segment the image')
        self.segment_button.pack(side=LEFT, fill=X, expand=False)

    def refresh_ui(self):
        if len(self.history) > 0:
            self.canvas_bg = PhotoImage(file=self.history[-1])
            self.width, self.height = self.canvas_bg.width(), self.canvas_bg.height()
            self.canvas.config(width=self.width, height=self.height)
            self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)
        else:
            self.canvas.delete("all")

        self.update_controls()

    def update_controls(self):
        self.prompt["state"] = DISABLED
        self.prompt['bg']    = '#D3D3D3'

    def update_canvas_image(self, image):
        self.history.append('history/{}.png'.format(time.time()))
        image.save(self.history[-1])
        self.refresh_ui()

    def segment(self):
        if DEBUG:
            self.segment_thread()
        else:
            threading.Thread(target=self.segment_thread).start()

    def segment_thread(self):
        # Get all necessary arguments from UI
        prompt     = self.prompt.get('1.0', 'end-1 chars')
        model_name = self.checkpoint.get()

        if len(self.history) > 0:
            init_image = load_image(self.history[-1])
        else:
            # If no image use all black (noise didn't work as well *np.random.randint(0, 255, (self.height, self.width, 3), "uint8")*)
            init_image = Image.fromarray(np.zeros((self.width, self.height, 3), 'uint8'))

        generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
        outputs = generator(init_image, points_per_batch=64)

        masks = outputs["masks"]
        image = show_masks_on_image(self.history[-1], masks)
        self.update_canvas_image(image)

        # Use to validate inputs and outputs
        if DEBUG:
            print(prompt)
            print(model_name)


