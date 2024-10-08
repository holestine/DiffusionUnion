from tkinter import *
from tkinter import filedialog, messagebox
from idlelib.tooltip import Hovertip
from diffusers.utils import load_image, make_image_grid
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import threading
from diffusers import AutoPipelineForInpainting, LDMSuperResolutionPipeline
import torch
import time
import numpy as np
from controls import create_number_control, create_toolbar_button

DEBUG = False

class inpainting_ui:

    def __init__(self, parent, history, width=512, height=512):

        # Efficient attention is not native in old PyTorch versions and is needed to reduce GPU usage
        self.use_efficient_attention = int(torch.__version__.split('.')[0]) < 2

        # Size of image to work with
        self.width = width
        self.height = height

        # Images created this session
        self.history = history

        # Get frames needed for layout
        toolbar, left_frame, right_frame = self.create_layout(parent)

        # Populate controls
        self.initialize_toolbar(toolbar)
        self.initialize_prompts(right_frame)
        self.initialize_canvas(left_frame)
        self.update_controls()

        # State variables
        self.drawing = False
        self.current_mask = []

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
        # Create canvas and wire up events
        # See https://python-course.eu/tkinter/events-and-binds-in-tkinter.php for binding info
        self.canvas = Canvas(parent, bg="black", width=self.width, height=self.height)
        self.canvas.pack(fill=BOTH, expand=False)
        self.canvas.bind("<Button-1>",        self.start_drawing)
        self.canvas.bind("<B1-Motion>",       self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Create a new mask to draw on
        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)

    def initialize_prompts(self, parent):
        # Create text box for entering the prompt
        prompt = "a photograph of an alien space ship with a metallic and mechanical appearance hovering above the ground in a highly forested area in the arctic near a frozen waterfall and rocky cliffs, highly detailed, 8k, realistic, stars in the sky, colorful"
        Label(parent, text="Positive Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.prompt = Text(parent, height=1, wrap=WORD)
        self.prompt.insert(END, prompt)
        self.prompt.pack(side=TOP, fill=BOTH, expand=True)

        # Create text box for entering negative prompt
        Label(parent, text="Negative Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.negative_prompt = Text(parent, height=1, wrap=WORD)
        self.negative_prompt.insert(END, "bad anatomy, deformed, ugly, poor details, blurry")
        self.negative_prompt.pack(side=TOP, fill=BOTH, expand=True)

    def initialize_toolbar(self, toolbar):
        
        # Create combo box for selecting a diffusion model
        checkpoint_frame = Frame(toolbar, bg='grey')
        checkpoint_options = ["Stable Diffusion 1.5", "Stable Diffusion XL 1.5", "Kandinsky 2.2"]
        self.checkpoint = StringVar(checkpoint_frame, checkpoint_options[0])
        Hovertip(checkpoint_frame, 'Select the diffusion model to use')
        Label(checkpoint_frame, text="Model", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        checkpoint_menu = OptionMenu(checkpoint_frame, self.checkpoint, *checkpoint_options)
        checkpoint_menu.config(width=20)
        checkpoint_menu.pack(side=LEFT, fill=X, expand=True)
        checkpoint_frame.pack(side=LEFT, fill=X, expand=False)
        self.checkpoint.trace_add("write", self.checkpoint_selection_callback) # Need to update UI when this changes

        # Create a control for entering the brush size
        self.radius_entry = create_number_control(toolbar, 100, 'Brush Radius', 'Enter the radius of the brush used to draw the mask', min=1)

        # Create a control for entering the mask blur
        self.blur_entry = create_number_control(toolbar, 33, 'Blur Factor', 'Amount for the mask to blend with the original image.', min=0)

        # Create a control for entering the generator value
        self.generator_entry = create_number_control(toolbar, 1, 'Generator', 'Different int values produce different results.', min=1)

        # Create a control for entering the generator value
        self.inference_steps_entry = create_number_control(toolbar, 200, 'Inference Steps', 'Higher values produce better images but take longer.', min=1, max=999)

        # Create textbox for entering the guidance value
        #self.guidance_entry = create_number_control(toolbar, 7.5, "Guidance", 'Enter a numeric value. Values between 7 and 8.5 are usually good choices, the default is 7.5. Higher values should make the image more closely match the prompt.')

        # Create a button to generate the image
        self.generate_button = create_toolbar_button(toolbar, "Generate", self.generate, 'Generate a new image')

        # Create a button to clear the canvas
        self.clear_button = create_toolbar_button(toolbar, "Clear Mask", self.clear_mask, 'Clear the current mask')

        # Create a button to increase the image's resolution
        self.super_res_button = create_toolbar_button(toolbar, "Super Res", self.super_res, 'Increase the image resolution')

        # Create a button to revert changes
        self.undo_button = create_toolbar_button(toolbar, "Undo", self.undo, 'Undo the last generated image', RIGHT)
        
    def checkpoint_selection_callback(self, *args):
        self.update_controls()
        
    def update_controls(self):
        checkpoint = self.checkpoint.get()
        if checkpoint == "Stable Diffusion 1.5" or checkpoint == "Stable Diffusion XL 1.5":
            self.negative_prompt['state'] = DISABLED
            self.negative_prompt['bg'] = '#D3D3D3'
            self.radius_entry['state'] = NORMAL
            self.blur_entry['state'] = NORMAL
            self.generator_entry['state'] = NORMAL
            self.inference_steps_entry['state'] = NORMAL
        elif checkpoint == "Stable Diffusion 2.1":
            self.negative_prompt['state'] = DISABLED
            self.negative_prompt['bg'] = '#D3D3D3'
            self.radius_entry['state'] = DISABLED
            self.blur_entry['state'] = DISABLED
            self.generator_entry['state'] = DISABLED
            self.inference_steps_entry['state'] = DISABLED
        elif checkpoint == "Kandinsky 2.2":
            self.negative_prompt["state"] = NORMAL
            self.negative_prompt['bg'] = '#FFFFFF'
            self.radius_entry['state'] = NORMAL
            self.blur_entry['state'] = DISABLED
            self.generator_entry['state'] = DISABLED
            self.inference_steps_entry['state'] = DISABLED

        if len(self.history) >= 1:
            self.undo_button["state"] = NORMAL
        else:
            self.undo_button["state"] = DISABLED

    def clear_mask(self):
        self.current_mask = []

        # Create a new mask to draw on
        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)
        self.refresh_ui()

    def refresh_ui(self):

        if len(self.history) > 0:
            self.canvas.delete("all")
            self.canvas_bg = PhotoImage(file=self.history[-1])
            self.width, self.height = self.canvas_bg.width(), self.canvas_bg.height()
            self.canvas.config(width=self.width, height=self.height)
            self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)
        else:
            self.canvas.delete("all")

        for (left, top, right, bottom) in self.current_mask:
            self.canvas.create_oval(left, top, right, bottom, fill="white", outline="")

        self.update_controls()

    def update_canvas_image(self, image):
        self.history.append('history/{}.png'.format(time.time()))
        image.save(self.history[-1])
        self.refresh_ui()

    def undo(self):
        # Create new background from previous saved file
        self.history.pop()
        self.refresh_ui()     
    
    def start_drawing(self, event):
        self.drawing = True

    def draw(self, event):
        # Draw a circle on the image and on the mask in the exact same location
        if self.drawing:
            x, y = event.x, event.y
            radius = int(self.radius_entry.get())
            left, top, right, bottom = x-radius, y-radius, x+radius, y+radius
            self.canvas.create_oval(left, top, right, bottom, fill="white", outline="")
            # This draws the same oval on self.mask
            self.mask_editor.ellipse([(left, top), (right, bottom)], fill=(255))
            # No easy way to apply masks with tkinter so keep track of how it's created
            self.current_mask.append((left, top, right, bottom))
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
        prompt              = self.prompt.get(         '1.0', 'end-1 chars')
        negative_prompt     = self.negative_prompt.get('1.0', 'end-1 chars')
        generator           = int(self.generator_entry.get())
        blur_factor         = int(self.blur_entry.get())
        num_inference_steps = int(self.inference_steps_entry.get())
        model_name = self.checkpoint.get()

        if len(self.history) > 0:
            init_image = load_image(self.history[-1])
        else:
            # I no image use all black (noise didn't work as well *np.random.randint(0, 255, (self.height, self.width, 3), "uint8")*)
            init_image = Image.fromarray(np.zeros((self.width, self.height, 3), 'uint8'))

        if model_name == "Stable Diffusion 1.5":
            pipe = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
            )
            pipe.enable_model_cpu_offload()
            if self.use_efficient_attention:
                pipe.enable_xformers_memory_efficient_attention()
            self.mask = pipe.mask_processor.blur(self.mask, blur_factor=blur_factor)
            generator = torch.Generator("cuda").manual_seed(generator)

            image = pipe(prompt=prompt, 
                         image=init_image, 
                         mask_image=self.mask, 
                         generator=generator, 
                         num_inference_steps=num_inference_steps, 
                         #strength=float(self.strength_entry.get()),
                         #guidance_scale=float(self.guidance_entry.get())
            ).images[0]
        elif model_name == "Stable Diffusion XL 1.5":
            pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
            )
            pipe.enable_model_cpu_offload()
            if self.use_efficient_attention:
                pipe.enable_xformers_memory_efficient_attention()
            self.mask = pipe.mask_processor.blur(self.mask, blur_factor=blur_factor)
            generator = torch.Generator("cuda").manual_seed(generator)
            
            image = pipe(prompt=prompt, 
                         image=init_image, 
                         mask_image=self.mask, 
                         generator=generator, 
                         num_inference_steps=num_inference_steps).images[0]
        elif model_name == "Kandinsky 2.2":
            pipe = AutoPipelineForInpainting.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
            )
            pipe.enable_model_cpu_offload()
            if self.use_efficient_attention:
                pipe.enable_xformers_memory_efficient_attention()

            image = pipe(prompt=prompt, 
                         negative_prompt=negative_prompt, 
                         image=init_image, 
                         mask_image=self.mask).images[0]
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

        self.clear_mask()
        self.update_canvas_image(image)
