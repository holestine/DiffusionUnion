from tkinter import *
from tkinter import filedialog
from idlelib.tooltip import Hovertip
import threading
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusion3Pipeline
from diffusers import FluxPipeline, DPMSolverMultistepScheduler, KolorsPipeline, DiffusionPipeline, EDMDPMSolverMultistepScheduler
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import time
from controls import create_toolbar_button
from huggingface_hub import login
from private import hugging_face_token
from transformers import pipeline
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


DEBUG = False

class image_generation_ui:

    def __init__(self, parent, history, width=512, height=512):

        # Efficient attention is not native in old PyTorch versions and is needed to reduce GPU usage
        self.use_efficient_attention = int(torch.__version__.split('.')[0]) < 2

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Login into HuggingFace to use Stable Diffusion 3
        try:
            login(hugging_face_token) # had to get a READ token from https://huggingface.co/settings/tokens/new
        except:
            print("Hugging face login failed")

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
        prompt = "Inside a Victorian-era laboratory filled with steampunk gadgets and machinery. A scientist in a leather apron and goggles works on a complex contraption made of brass, gears, and glass tubes filled with glowing liquids. The room is illuminated by warm, flickering gas lamps, and in the background, a large clockwork mechanism slowly turns, powering the various devices scattered around the room."
        Label(parent, text="Positive Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.prompt = Text(parent, height=1, wrap=WORD)
        self.prompt.insert(END, prompt)
        self.prompt.pack(side=TOP, fill=BOTH, expand=True)

    def initialize_toolbar(self, toolbar):
        
        # Create combo box for selecting a diffusion model
        checkpoint_frame = Frame(toolbar, bg='grey')
        checkpoint_options = ["Stable Diffusion 2.1", "Stable Diffusion 3", "FLUX.1-dev", "Kolors", "Playground-v2.5"]
        self.checkpoint = StringVar(checkpoint_frame, checkpoint_options[0])
        Hovertip(checkpoint_frame, 'Select the model to use')
        Label(checkpoint_frame, text="Model", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        checkpoint_menu = OptionMenu(checkpoint_frame, self.checkpoint, *checkpoint_options)
        checkpoint_menu.config(width=20)
        checkpoint_menu.pack(side=LEFT, fill=X, expand=True)
        checkpoint_frame.pack(side=LEFT, fill=X, expand=False)

        # Create a button to load an image
        self.load_button = create_toolbar_button(toolbar, 'Load Image', self.load_background, 'Open an image')

        # Create a button to generate the image
        self.generate_button = create_toolbar_button(toolbar, 'Generate Image', self.generate, 'Generate a new image')

        # Create a button to generate a chrome ball
        self.chrome_ball_button = create_toolbar_button(toolbar, 'Generate Chrome Ball', self.create_chrome_ball, 'Generate a chrome ball for a light probe')

        # Create a button to revert changes
        self.undo_button = create_toolbar_button(toolbar, 'Undo', self.undo, 'Undo the last generated image')

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
        if len(self.history) >= 1:
            self.undo_button["state"] = NORMAL
        else:
            self.undo_button["state"] = DISABLED

    def update_canvas_image(self, image):
        self.history.append('history/{}.png'.format(time.time()))
        image.save(self.history[-1])
        self.refresh_ui()

    def undo(self):
        # Create new background from previous saved file
        self.history.pop()
        self.refresh_ui()

    def generate(self):
        if DEBUG:
            self.generate_thread()
        else:
            threading.Thread(target=self.generate_thread).start()

    def generate_thread(self):
        # Get all necessary arguments from UI
        prompt     = self.prompt.get('1.0', 'end-1 chars')
        model_name = self.checkpoint.get()

        if model_name == "Stable Diffusion 2.1":
            # https://huggingface.co/stabilityai/stable-diffusion-2-1
            pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to(self.device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            image = pipe(prompt=prompt).images[0]
        elif model_name == "Stable Diffusion 3":
            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to(self.device)
            image = pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
            ).images[0]
        elif model_name == "FLUX.1-dev":
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

            image = pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
        elif model_name == "Kolors":
            pipe = KolorsPipeline.from_pretrained("Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16").to(self.device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

            image = pipe(
                prompt,
                negative_prompt="",
                guidance_scale=6.5,
                num_inference_steps=25,
            ).images[0]
        elif model_name == "Playground-v2.5":
            pipe = DiffusionPipeline.from_pretrained(
                "playgroundai/playground-v2.5-1024px-aesthetic",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(self.device)

            # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
            pipe.scheduler = EDMDPMSolverMultistepScheduler()

            image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]
        else:
            print("Specify a supported model.\n")
            return

        # Use to validate inputs and outputs
        if DEBUG:
            print(prompt)
            print(model_name)

        self.update_canvas_image(image)
        
    def create_chrome_ball(self):

        # Configuration
        IS_UNDER_EXPOSURE = False #change this option for output underexposured ball 
        if IS_UNDER_EXPOSURE:
            PROMPT = "a perfect black dark mirrored reflective chrome ball sphere"
        else:
            PROMPT = "a perfect mirrored reflective chrome ball sphere"

        NEGATIVE_PROMPT = "matte, diffuse, flat, dull"

        # load pipeline
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        ).to("cuda")
        pipe.load_lora_weights("DiffusionLight/DiffusionLight")
        pipe.fuse_lora(lora_scale=0.75)
        depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")

        # prepare input image
        init_image = load_image(self.history[-1])
        depth_image = depth_estimator(images=init_image)['depth']

        # create mask and depth map with mask for inpainting
        def get_circle_mask(size=256):
            x = torch.linspace(-1, 1, size)
            y = torch.linspace(1, -1, size)
            y, x = torch.meshgrid(y, x)
            z = (1 - x**2 - y**2)
            mask = z >= 0
            return mask 
        
        # Get the mask for the chrome ball
        ball_size = 100
        mask = get_circle_mask(size=ball_size).numpy()
        depth = np.asarray(depth_image).copy()
        top = int(depth.shape[1]/2 - int(ball_size/2))
        left = int(depth.shape[0]/2 - int(ball_size/2))
        depth[left:left+ball_size, top:top+ball_size] = depth[left:left+ball_size, top:top+ball_size] * (1 - mask) + (mask * 255)
        depth_mask = Image.fromarray(depth)
        mask_image = np.zeros_like(depth)
        mask_image[left:left+ball_size, top:top+ball_size] = mask * 255
        mask_image = Image.fromarray(mask_image)

        #plt.imshow(depth_mask)
        #plt.show()

        # run the pipeline
        output = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=30,
            image=init_image,
            mask_image=mask_image,
            control_image=depth_mask,
            controlnet_conditioning_scale=0.5,
        )

        image = output["images"][0]
        self.update_canvas_image(image)

    def load_background(self):
        res = filedialog.askopenfile(initialdir="./history")
        if res:
            self.history.append(res.name)
            self.refresh_ui()
            self.update_controls()
