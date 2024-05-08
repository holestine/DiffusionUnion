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

    def __init__(self, parent, width=512, height=512):
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
        self.initialize_canvas(left_frame)
        self.initialize_prompts(right_frame)
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

        # Make black background
        self.update_canvas_image(Image.fromarray(np.zeros((self.width, self.height, 3), 'uint8')))

    def initialize_prompts(self, parent):
        # Create text box for entering the prompt
        Label(parent, text="Positive Prompt:", anchor=W).pack(side=TOP, fill=X, expand=FALSE)
        self.prompt = Text(parent, height=3, wrap=WORD)
        self.prompt.insert(END, "a clandestine mystical city in the mountains surrounded by wildlife, bushes and trees near a waterfall and next to a rocky river during a lightning storm with a clear starry ski in the background, highly detailed, 8k, in a style similar to Tim Burton, realistic level of detail")
        self.prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

        # Create text box for entering negative prompt
        Label(parent, text="Negative Prompt:", anchor=W).pack(side=TOP, fill=X, expand=FALSE)
        self.negative_prompt = Text(parent, height=3, wrap=WORD)
        self.negative_prompt.insert(END, "bad anatomy, deformed, ugly, normal")
        self.negative_prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

    def initialize_toolbar(self, toolbar):
        
        # Create combo box for entering radius
        radius_frame = Frame(toolbar, bg='grey')
        Hovertip(radius_frame,'Select the radius for the paintbrush used to draw the mask')
        radius_frame.pack(side=LEFT, fill=X, expand=FALSE)

        Label(radius_frame, text="Radius:", anchor=W).pack(side=LEFT, fill=Y, expand=FALSE)

        radius_options = [1,2,3,4,5,10,15,20,50,100]
        self.radius = IntVar()
        self.radius.set(radius_options[-1])        
        OptionMenu(radius_frame, self.radius, *radius_options).pack(side=LEFT, fill=X, expand=TRUE)

        # Create combo box for entering the mask blur
        blur_frame = Frame(toolbar, bg='grey')
        Hovertip(blur_frame,'Select the blur factor for the mask')
        blur_frame.pack(side=LEFT, fill=X, expand=FALSE)

        Label(radius_frame, text="Blur Factor:", anchor=W).pack(side=LEFT, fill=Y, expand=FALSE)

        blur_options = [0,1,2,3,4,5,10,15,20,50,100]
        self.blur = IntVar()
        self.blur.set(blur_options[0])
        
        self.blur_options = OptionMenu(blur_frame, self.blur, *blur_options)
        self.blur_options.pack(side=LEFT, fill=X, expand=TRUE)

        # Create combo box for selecting diffusion model
        model_frame = Frame(toolbar, bg='grey')
        Hovertip(model_frame,'Select the diffusion model to use')
        model_frame.pack(side=LEFT, fill=X, expand=FALSE)

        Label(model_frame, text="Diffusion Model:", anchor=W).pack(side=LEFT, fill=Y, expand=FALSE)

        model_options = ["Stable Diffusion", "Stable Diffusion XL", "Kandinsky 2.2"]
        self.checkpoint = StringVar(model_frame, "Kandinsky 2.2")
        self.checkpoint.trace_add("write", self.checkpoint_selection_callback)
        
        OptionMenu(model_frame, self.checkpoint, *model_options).pack(side=LEFT, fill=X, expand=TRUE)

        # Create a button to load a background
        self.load_button = Button(toolbar, text="Load Background", command=self.load_background)
        Hovertip(self.load_button,'Open an image')
        self.load_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create a button to generate the image
        self.generate_button = Button(toolbar, text="Generate Image", command=self.generate)
        Hovertip(self.generate_button,'Generate a new image')
        self.generate_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create a button to clear canvas
        self.clear_button = Button(toolbar, text="Clear Mask", command=self.clear_mask)
        Hovertip(self.clear_button,'Clear the current mask')
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
            self.blur_options['state'] = NORMAL
        elif checkpoint == "Kandinsky 2.2":
            self.negative_prompt["state"] = NORMAL
            self.negative_prompt['bg'] = '#FFFFFF'
            self.blur_options['state'] = DISABLED
            
        if len(self.history) > 1:
            self.undo_button["state"] = NORMAL
        else:
            self.undo_button["state"] = DISABLED

    def undo(self):
        self.history.pop()
        self.canvas_bg = PhotoImage(file=self.history[-1])

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)

        self.update_controls()
    
    def start_drawing(self, event):
        self.drawing = TRUE

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            radius = self.radius.get()
            left, top, right, bottom = x-radius, y-radius, x+radius, y+radius
            self.canvas.create_oval(left, top, right, bottom, fill="white", outline="")
            self.mask_editor.ellipse([(left, top), (right, bottom)], fill=(255))
            return

    def stop_drawing(self, event):
        self.drawing = False

    def clear_mask(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

    def super_res(self):
        threading.Thread(target=self.super_res_thread).start()

    def super_res_thread(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "CompVis/ldm-super-resolution-4x-openimages"

        # load model and scheduler
        pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
        pipeline = pipeline.to(device)

        # let's download an  image
        low_res_img = Image.open(self.history[-1]).convert("RGB")

        try:
            upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
            self.update_canvas_image(upscaled_image)
        except Exception as ex:
            print(ex)
            messagebox.showinfo("Error", ex.args[0]) 

    
    def generate(self):
        threading.Thread(target=self.generate_thread).start()

    def generate_thread(self):
        # Get all necessary arguments from UI
        prompt = self.prompt.get('1.0', 'end-1 chars')
        negative_prompt = self.negative_prompt.get('1.0', 'end-1 chars') 
        init_image = load_image(self.history[-1])
        model_name = self.checkpoint.get()

        if model_name == "Stable Diffusion":
            pipeline = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
            )
            pipeline.enable_model_cpu_offload()
            # Remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
            #pipeline.enable_xformers_memory_efficient_attention()
            self.mask = pipeline.mask_processor.blur(self.mask, blur_factor=33)
            generator = torch.Generator("cuda").manual_seed(92)
            image = pipeline(prompt=prompt, image=init_image, mask_image=self.mask, generator=generator).images[0]
        elif model_name == "Stable Diffusion XL":
            pipeline = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
            )
            pipeline.enable_model_cpu_offload()
            # Remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
            #pipeline.enable_xformers_memory_efficient_attention()
            generator = torch.Generator("cuda").manual_seed(92)
            self.mask = pipeline.mask_processor.blur(self.mask, blur_factor=33)
            image = pipeline(prompt=prompt, image=init_image, mask_image=self.mask, generator=generator).images[0]
        elif model_name == "Kandinsky 2.2":
            pipeline = AutoPipelineForInpainting.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
            )
            pipeline.enable_model_cpu_offload()
            # Remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
            #pipeline.enable_xformers_memory_efficient_attention()
            image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=self.mask).images[0]
        else:
            print("Specify a supported model.\n")
            return
        
        self.update_canvas_image(image)

        # Use to validate inputs and outputs
        if DEBUG:
            print(prompt)
            print(negative_prompt)
            print(model_name)
            plt.imshow(make_image_grid([init_image, self.mask, image], rows=1, cols=3))
            plt.show()
        

    def update_canvas_image(self, image):

        self.history.append('history/{}.png'.format(time.time()))
        image.save(self.history[-1])
        self.canvas_bg = PhotoImage(file=self.history[-1])
        
        if len(self.history) > 1:
            self.undo_button["state"] = NORMAL
        else:
            self.undo_button["state"] = DISABLED
        
        self.canvas.delete("all")
        self.width, self.height = self.canvas_bg.width(), self.canvas_bg.height()
        self.canvas.config(width=self.width, height=self.height)
        self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)


    def load_background(self):
        res = filedialog.askopenfile(initialdir="./history")
        if res:
            self.canvas.delete("all")
            self.history.append(res.name)
            self.canvas_bg = PhotoImage(file=self.history[-1])
            self.width, self.height = self.canvas_bg.width(), self.canvas_bg.height()
            self.canvas.config(width=self.width, height=self.height)
            self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

            self.mask = Image.new("L", (self.width, self.height))
            self.mask_editor = ImageDraw.Draw(self.mask)
