from tkinter import *
from tkinter import filedialog
from diffusers.utils import load_image, make_image_grid
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import threading
from diffusers import AutoPipelineForInpainting, LDMSuperResolutionPipeline
import torch
import os, time
import numpy as np

class inpainting_ui:

    def __init__(self, parent, width=512, height=512):
        self.debug = FALSE
        self.history = []
        self.width = width
        self.height = height
        toolbar, left_frame, right_frame = self.create_layout(parent)

        self.initialize_toolbar(toolbar)
        self.initialize_canvas(left_frame)
        self.initialize_prompts(right_frame)


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

    def initialize_canvas(self, left_frame):
        # Create a 512x512 black image to start with 
        self.history.append('history/{}.png'.format(time.time()))
        Image.fromarray(np.zeros((self.width,self.height,3), 'uint8')).save(self.history[-1])

        self.canvas_bg = PhotoImage(file=self.history[-1])
        self.width, self.height = self.canvas_bg.width(), self.canvas_bg.height()
        

        # Create canvas for drawing mask
        self.canvas = Canvas(left_frame, bg="black", width=self.width, height=self.height)
        self.canvas.pack(fill=BOTH, expand=False)
        self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

        # Create a grayscale image the same size as the canvas to track mask creation
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)

        # Add events for left mouse button on canvas. See https://python-course.eu/tkinter/events-and-binds-in-tkinter.php
        self.canvas.bind("<Button-1>",        self.start_drawing)
        self.canvas.bind("<B1-Motion>",       self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.drawing = False

    def initialize_prompts(self, right_frame):
        # Create text box for entering the prompt
        self.prompt_label = Label(right_frame, text="Positive Prompt:", anchor=W)
        self.prompt_label.pack(side=TOP, fill=X, expand=FALSE)
        
        self.prompt = Text(right_frame, height=3, wrap=WORD)
        self.prompt.insert(END, "a clandestine mystical city in the mountains surrounded by wildlife, bushes and trees near a waterfall and next to a rocky river during a lightning storm with a clear starry ski in the background, highly detailed, 8k, in a style similar to Tim Burton, realistic level of detail")
        self.prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

        # Create text box for entering negative prompt
        self.negative_prompt_label = Label(right_frame, text="Negative Prompt:", anchor=W)
        self.negative_prompt_label.pack(side=TOP, fill=X, expand=FALSE)
        
        self.negative_prompt = Text(right_frame, height=3, wrap=WORD)
        self.negative_prompt.insert(END, "bad anatomy, deformed, ugly, normal")
        self.negative_prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

    def initialize_toolbar(self, toolbar):
        
        # Create combo box for entering radius
        radius_frame = Frame(toolbar, bg='grey')
        radius_frame.pack(side=LEFT, fill=X, expand=FALSE)

        self.radius_label = Label(radius_frame, text="Radius:", anchor=W)
        self.radius_label.pack(side=LEFT, fill=Y, expand=FALSE)

        self.radius = IntVar()
        self.radius.set('20')
        options = [1,2,3,4,5,10,15,20,50,100]
        drop = OptionMenu(radius_frame, self.radius, *options)
        drop.pack(side=LEFT, fill=X, expand=TRUE)

        # Create combo box for selecting diffusion model
        model_frame = Frame(toolbar, bg='grey')
        model_frame.pack(side=LEFT, fill=X, expand=FALSE)

        self.model_label = Label(model_frame, text="Diffusion Model:", anchor=W)
        self.model_label.pack(side=LEFT, fill=Y, expand=FALSE)

        self.model = StringVar(model_frame, "Kandinsky 2.2")

        model_options = ["Stable Diffusion", "Stable Diffusion XL", "Kandinsky 2.2"]
        model_drop = OptionMenu(model_frame, self.model, *model_options)
        model_drop.pack(side=LEFT, fill=X, expand=TRUE)

        # Create button to load a background
        self.load_button = Button(toolbar, text="Load Background", command=self.load_background)
        self.load_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create button to generate the image
        self.clear_button = Button(toolbar, text="Generate Image", command=self.generate)
        self.clear_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create a button to clear canvas
        self.generate_button = Button(toolbar, text="Clear Mask", command=self.clear_mask)
        self.generate_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create a button to increase the image's resolution
        self.generate_button = Button(toolbar, text="Super Res", command=self.super_res)
        self.generate_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create a button to revert the changes
        self.undo_button = Button(toolbar, text="Undo", command=self.undo)
        self.undo_button.pack(side=LEFT, fill=X, expand=FALSE)
        self.undo_button["state"] = DISABLED

    def undo(self):
        self.history.pop()
        self.canvas_bg = PhotoImage(file=self.history[-1])

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)

        if len(self.history) > 1:
            self.undo_button["state"] = NORMAL
        else:
            self.undo_button["state"] = DISABLED
    
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "CompVis/ldm-super-resolution-4x-openimages"

        # load model and scheduler
        pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
        pipeline = pipeline.to(device)

        # let's download an  image
        low_res_img = Image.open(self.history[-1]).convert("RGB")
        low_res_img = low_res_img.resize((128, 128))

        # run pipeline in inference (sample random noise and denoise)
        upscaled_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
        # save image
        upscaled_image.save('history/{}.png'.format(time.time()))
    
    def generate(self):
        threading.Thread(target=self.generate_thread).start()

    def generate_thread(self):
        # Get all necessary arguments from UI
        prompt = self.prompt.get('1.0', 'end-1 chars')
        negative_prompt = self.negative_prompt.get('1.0', 'end-1 chars') 
        init_image = load_image(self.history[-1])
        model_name = self.model.get()

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
        
        # Use to validate inputs and outputs
        if self.debug:
            print(prompt)
            print(negative_prompt)
            print(model_name)
            plt.imshow(make_image_grid([init_image, self.mask, image], rows=1, cols=3))
            plt.show()

        if not os.path.exists('history'):
            os.makedirs('history')

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
            self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)
