from tkinter import *
from tkinter import filedialog, messagebox
from diffusers.utils import load_image, make_image_grid
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import threading
from diffusers import AutoPipelineForInpainting, LDMSuperResolutionPipeline
import torch
import os, time
import numpy as np
from idlelib.tooltip import Hovertip

DEBUG = FALSE

class inpainting_ui:

    def filter_numbers_only(self, value):
        if str.isdigit(value) or value == "":
            return True
        else:
            return False

    def __init__(self, parent, width=512, height=512):

        # Efficient attention is not native in old PyTorch versions and is needed to reduce GPU usage
        self.use_efficient_attention = int(torch.__version__.split('.')[0]) < 2

        # Size of image to work with
        self.width = width
        self.height = height

        # Used to store generated images
        self.history = []
        if not os.path.exists('history'):
            os.makedirs('history')

        # Get frames needed for layout
        toolbar, left_frame, right_frame = self.create_layout(parent)

        # Populate controls
        self.initialize_toolbar(toolbar)
        self.initialize_prompts(right_frame)
        self.initialize_canvas(left_frame)
        self.update_controls()

        # State variables
        self.drawing = False

    def create_layout(self, parent):
        # Create toolbar
        toolbar = Frame(parent, width=2*self.width, height=20, bg='grey')
        toolbar.pack(side=TOP, fill=X, expand=FALSE)

        # Create left frame
        left_frame = Frame(parent, width=self.width, height=self.height, bg='grey')
        left_frame.pack(side=LEFT, fill=BOTH, expand=FALSE)

        # Create right frame
        right_frame = Frame(parent, width=self.width, height=self.height, bg='grey')
        right_frame.pack(side=RIGHT, fill=BOTH, expand=TRUE)

        return toolbar, left_frame, right_frame

    def initialize_canvas(self, parent):
        # Create canvas and wire up events
        # See https://python-course.eu/tkinter/events-and-binds-in-tkinter.php for binding info
        self.canvas = Canvas(parent, bg="black", width=self.width, height=self.height)
        self.canvas.pack(fill=BOTH, expand=False)
        self.canvas.bind("<Button-1>",        self.start_drawing)
        self.canvas.bind("<B1-Motion>",       self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Make black background (tried noise but it didn't work as well (np.random.randint(0, 255, (self.height, self.width, 3), "uint8")))
        self.update_canvas_image(Image.fromarray(np.zeros((self.width, self.height, 3), 'uint8')))

    def initialize_prompts(self, parent):
        # Create text box for entering the prompt
        prompt = "a mountain town next to a ski resort surrounded by bushes and trees near a waterfall and a rocky river during a lightning storm with a clear starry sky in the background, highly detailed, 8k, realistic"
        Label(parent, text="Positive Prompt:", anchor=W).pack(side=TOP, fill=X, expand=FALSE)
        self.prompt = Text(parent, height=1, wrap=WORD)
        self.prompt.insert(END, prompt)
        self.prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

        # Create text box for entering negative prompt
        Label(parent, text="Negative Prompt:", anchor=W).pack(side=TOP, fill=X, expand=FALSE)
        self.negative_prompt = Text(parent, height=1, wrap=WORD)
        self.negative_prompt.insert(END, "bad anatomy, deformed, ugly")
        self.negative_prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

    def initialize_toolbar(self, toolbar):
        
        # Create combo box for selecting diffusion model
        checkpoint_frame = Frame(toolbar, bg='grey')
        checkpoint_options = ["Stable Diffusion", "Stable Diffusion XL", "Kandinsky 2.2"]
        self.checkpoint = StringVar(checkpoint_frame, checkpoint_options[0])
        Hovertip(checkpoint_frame, 'Select the diffusion model to use')
        Label(checkpoint_frame, text="Diffusion Model:", anchor=W).pack(side=LEFT, fill=Y, expand=FALSE)
        OptionMenu(checkpoint_frame, self.checkpoint, *checkpoint_options).pack(side=LEFT, fill=X, expand=TRUE)
        checkpoint_frame.pack(side=LEFT, fill=X, expand=FALSE)
        self.checkpoint.trace_add("write", self.checkpoint_selection_callback) # Need to update UI when this changes

        # Create textbox for entering radius
        radius_frame = Frame(toolbar, bg='grey')
        self.radius = IntVar(radius_frame, 100)
        Hovertip(radius_frame, 'Select the radius for the paintbrush used to draw the mask')
        Label(radius_frame, text="Radius:", anchor=W).pack(side=LEFT, fill=Y, expand=FALSE)
        self.radius_entry = Entry(radius_frame, validate='all', validatecommand=(toolbar.register(self.filter_numbers_only), '%P'), textvariable=self.radius)
        self.radius_entry.pack(side=LEFT, fill=BOTH, expand=TRUE)
        radius_frame.pack(side=LEFT, fill=BOTH, expand=FALSE)

        # Create textbox for entering the mask blur
        blur_frame = Frame(toolbar, bg='grey')
        self.blur_factor = IntVar(blur_frame, 33)
        Hovertip(blur_frame, 'Enter the blur factor for the mask')
        Label(blur_frame, text="Blur Factor:", anchor=W).pack(side=LEFT, fill=Y, expand=FALSE)
        self.blur_entry = Entry(blur_frame, validate='all', validatecommand=(toolbar.register(self.filter_numbers_only), '%P'), textvariable=self.blur_factor)
        self.blur_entry.pack(side=LEFT, fill=BOTH, expand=TRUE)
        blur_frame.pack(side=LEFT, fill=BOTH, expand=FALSE)

        # Create a button to load a background
        self.load_button = Button(toolbar, text="Load Background", command=self.load_background)
        Hovertip(self.load_button, 'Open an image')
        self.load_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create a button to generate the image
        self.generate_button = Button(toolbar, text="Generate Image", command=self.generate)
        Hovertip(self.generate_button, 'Generate a new image')
        self.generate_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create a button to clear the canvas
        self.clear_button = Button(toolbar, text="Clear Mask", command=self.refresh_canvas)
        Hovertip(self.clear_button, 'Clear the current mask')
        self.clear_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create a button to increase the image's resolution
        self.super_res_button = Button(toolbar, text="Super Res", command=self.super_res)
        Hovertip(self.super_res_button,'Increase the image resolution')
        self.super_res_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create a button to revert changes
        self.undo_button = Button(toolbar, text="Undo", command=self.undo)
        Hovertip(self.undo_button,'Undo the last generated image')
        self.undo_button.pack(side=LEFT, fill=X, expand=FALSE)
        
    def checkpoint_selection_callback(self, *args):
        self.update_controls()
        
    def update_controls(self):
        checkpoint = self.checkpoint.get()
        if checkpoint == "Stable Diffusion" or checkpoint == "Stable Diffusion XL":
            self.negative_prompt['state'] = DISABLED
            self.negative_prompt['bg'] = '#D3D3D3'
            self.blur_entry['state'] = NORMAL
        elif checkpoint == "Kandinsky 2.2":
            self.negative_prompt["state"] = NORMAL
            self.negative_prompt['bg'] = '#FFFFFF'
            self.blur_entry['state'] = DISABLED

        if len(self.history) > 1:
            self.undo_button["state"] = NORMAL
        else:
            self.undo_button["state"] = DISABLED

    def refresh_canvas(self):
        self.canvas.delete("all")
        self.canvas_bg = PhotoImage(file=self.history[-1])
        self.width, self.height = self.canvas_bg.width(), self.canvas_bg.height()
        self.canvas.config(width=self.width, height=self.height)
        self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

        # Create a new mask to draw on
        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)

        self.update_controls()

    def update_canvas_image(self, image):
        self.history.append('history/{}.png'.format(time.time()))
        image.save(self.history[-1])
        self.refresh_canvas()

    def undo(self):
        # Create new background from previous saved file
        self.history.pop()
        self.refresh_canvas()     
    
    def start_drawing(self, event):
        self.drawing = TRUE

    def draw(self, event):
        # Draw a circle on the image and on the mask in the exact same location
        if self.drawing:
            x, y = event.x, event.y
            radius = self.radius.get()
            left, top, right, bottom = x-radius, y-radius, x+radius, y+radius
            self.canvas.create_oval(left, top, right, bottom, fill="white", outline="")
            self.mask_editor.ellipse([(left, top), (right, bottom)], fill=(255))
            return

    def stop_drawing(self, event):
        self.drawing = False

    def super_res(self):
        if DEBUG:
            self.super_res_thread()
        else:
            threading.Thread(target=self.super_res_thread).start()

    def super_res_thread(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "CompVis/ldm-super-resolution-4x-openimages"

        # load model and scheduler
        pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
        pipeline = pipeline.to(device)

        # Get the current image, may be able to pull this from self.canvas
        low_res_img = Image.open(self.history[-1]).convert("RGB")

        # Determine mask bounding box
        y, x = np.nonzero(self.mask)
        if len(x) > 0 and len(y) > 0:
            low_res_img = low_res_img.crop((min(x), min(y), max(x), max(y)))

        # Super Res the image
        try:
            super_res_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
            self.update_canvas_image(super_res_image)
        except Exception as ex:
            print(ex)
            messagebox.showinfo("Error", ex.args[0]) 

        # Use to validate inputs and outputs
        if DEBUG:
            plt.imshow(make_image_grid([low_res_img], rows=1, cols=1))
            plt.show()
    
    def generate(self):
        if DEBUG:
            self.generate_thread()
        else:
            threading.Thread(target=self.generate_thread).start()

    def generate_thread(self):
        # Get all necessary arguments from UI
        prompt          = self.prompt.get(         '1.0', 'end-1 chars')
        negative_prompt = self.negative_prompt.get('1.0', 'end-1 chars') 
        init_image = load_image(self.history[-1])
        model_name = self.checkpoint.get()

        if model_name == "Stable Diffusion":
            pipeline = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
            )
            pipeline.enable_model_cpu_offload()
            if self.use_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()
            self.mask = pipeline.mask_processor.blur(self.mask, blur_factor=self.blur_factor.get())
            generator = torch.Generator("cuda").manual_seed(92)
            image = pipeline(prompt=prompt, image=init_image, mask_image=self.mask, generator=generator).images[0]
        elif model_name == "Stable Diffusion XL":
            pipeline = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
            )
            pipeline.enable_model_cpu_offload()
            if self.use_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()
            generator = torch.Generator("cuda").manual_seed(92)
            self.mask = pipeline.mask_processor.blur(self.mask, blur_factor=self.blur_factor.get())
            image = pipeline(prompt=prompt, image=init_image, mask_image=self.mask, generator=generator).images[0]
        elif model_name == "Kandinsky 2.2":
            pipeline = AutoPipelineForInpainting.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
            )
            pipeline.enable_model_cpu_offload()
            if self.use_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()
            image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=self.mask).images[0]
        else:
            print("Specify a supported model.\n")
            return

        # Use to validate inputs and outputs
        if DEBUG:
            print(prompt)
            print(negative_prompt)
            print(model_name)
            plt.imshow(make_image_grid([init_image, self.mask, image], rows=1, cols=3))
            plt.show()

        self.update_canvas_image(image)
        
    def load_background(self):
        res = filedialog.askopenfile(initialdir="./history")
        if res:
            self.history.append(res.name)
            self.refresh_canvas()
