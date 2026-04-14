from tkinter import *
from tkinter import filedialog, ttk, messagebox
from idlelib.tooltip import Hovertip
import threading
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusion3Pipeline, StableDiffusionXLPipeline
from diffusers import FluxPipeline, DPMSolverMultistepScheduler, KolorsPipeline, DiffusionPipeline, EDMDPMSolverMultistepScheduler
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxControlPipeline
from diffusers.utils import load_image
import torch
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*upcast_vae.*")
warnings.filterwarnings("ignore", message=".*Siglip2ImageProcessorFast.*")
from controls import create_toolbar_button, create_number_control, load_display_image
from huggingface_hub import login
from private import hugging_face_token
from transformers import pipeline
from PIL import Image
from PIL import ImageTk
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
        prompt = "A painting of a landscape with an large old farmhouse growing some healthy crops with a dirt road winding through the scene. There is a thunder storm, lightning strikes nearby and the taller plants bend in the wind including a sunflower that is facing the camera.  The sun is showing through the clouds in the distance above some rugged mountains and a river flows nearby. Style of van gogh."
        Label(parent, text="Positive Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.prompt = Text(parent, height=1, wrap=WORD, pady=4)
        self.prompt.insert(END, prompt)
        self.prompt.pack(side=TOP, fill=BOTH, expand=True)

        Label(parent, text="Reference Image:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.ref_canvas = Canvas(parent, bg='#333333', height=200)
        self.ref_canvas.pack(side=TOP, fill=X, expand=False)

    def initialize_toolbar(self, toolbar):
        
        # Create combo box for selecting a diffusion model
        checkpoint_frame = Frame(toolbar, bg='grey')
        checkpoint_options = ["Stable Diffusion XL", "Stable Diffusion 3", "FLUX.1-dev", "FLUX.1-schnell", "FLUX.1-dev ControlNet Depth", "FLUX.1-dev IP-Adapter", "FLUX.1-Redux", "Kolors", "Playground-v2.5"]
        self.checkpoint = StringVar(checkpoint_frame, checkpoint_options[0])
        Hovertip(checkpoint_frame, 'Select the model to use')
        Label(checkpoint_frame, text="Model", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        checkpoint_menu = OptionMenu(checkpoint_frame, self.checkpoint, *checkpoint_options)
        checkpoint_menu.config(width=20)
        checkpoint_menu.pack(side=LEFT, fill=X, expand=True)
        checkpoint_frame.pack(side=LEFT, fill=X, expand=False)

        # Create combo box for selecting output resolution
        resolution_frame = Frame(toolbar, bg='grey')
        self.resolution_options = {
            "1024×1024":   (1024, 1024),
            "1920×1080":   (1920, 1080),
            "2560×1440":   (2560, 1440),
            "3840×2160":   (3840, 2160),
            "1080×1920":   (1080, 1920),
            "1280×720":    (1280, 720),
        }
        self.resolution = StringVar(resolution_frame, "1920×1080")
        Hovertip(resolution_frame, 'Select output resolution (all dimensions are multiples of 64)')
        Label(resolution_frame, text="Resolution", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        resolution_menu = OptionMenu(resolution_frame, self.resolution, *self.resolution_options.keys())
        resolution_menu.config(width=12)
        resolution_menu.pack(side=LEFT, fill=X, expand=True)
        resolution_frame.pack(side=LEFT, fill=X, expand=False)

        # Create a control for entering the generator seed
        self.generator_entry = create_number_control(toolbar, 0, 'Generator', 'Different int values produce different results.', min=0)

        # Create a button to load an image
        self.load_button = create_toolbar_button(toolbar, 'Load Image', self.load_background, 'Open an image')

        # Create a button to load a reference image (used by ControlNet, IP-Adapter, Redux)
        self.ref_image_path = None
        self.load_ref_button = create_toolbar_button(toolbar, 'Load Reference', self.load_reference, 'Load a reference image for ControlNet depth, IP-Adapter, or Redux')

        # Create a button to generate the image
        self.generate_button = create_toolbar_button(toolbar, 'Generate Image', self.generate, 'Generate a new image')

        # Create a button to generate a chrome ball
        #self.chrome_ball_button = create_toolbar_button(toolbar, 'Generate Chrome Ball', self.create_chrome_ball, 'Generate a chrome ball for a light probe')

        # Create a button to revert changes
        self.undo_button = create_toolbar_button(toolbar, 'Undo', self.undo, 'Undo the last generated image')

        # Progress bar — updated via callback_on_step_end during inference
        self.progress_var = DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(toolbar, variable=self.progress_var, maximum=100, length=200)
        self.progress_bar.pack(side=LEFT, padx=4)

    def refresh_ui(self):
        if len(self.history) > 0:
            load_display_image(self.canvas, self.history[-1])
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

    def _apply_offload(self, pipe, vram_needed_gb):
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_vram_gb >= vram_needed_gb:
            pipe.to(self.device)
        else:
            pipe.enable_sequential_cpu_offload()
            if hasattr(pipe, 'vae'):
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()

    def generate(self):
        if DEBUG:
            self.generate_thread()
        else:
            threading.Thread(target=self.generate_thread).start()

    def _make_progress_callback(self, num_steps):
        def callback(pipe, step, timestep, kwargs):
            self.progress_var.set((step + 1) / num_steps * 100)
            return kwargs
        return callback

    def generate_thread(self):
        # Get all necessary arguments from UI
        prompt         = self.prompt.get('1.0', 'end-1 chars')
        model_name     = self.checkpoint.get()
        generator_seed = int(self.generator_entry.get())
        width, height  = self.resolution_options[self.resolution.get()]

        self.progress_var.set(0)

        if model_name == "Stable Diffusion XL":
            steps = 30
            pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
            self._apply_offload(pipe, vram_needed_gb=12)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            image = pipe(prompt=prompt, height=height, width=width,
                         num_inference_steps=steps,
                         callback_on_step_end=self._make_progress_callback(steps)).images[0]
        elif model_name == "Stable Diffusion 3":
            steps = 28
            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
            self._apply_offload(pipe, vram_needed_gb=14)
            image = pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=steps,
                guidance_scale=7.0,
                height=height,
                width=width,
                callback_on_step_end=self._make_progress_callback(steps),
            ).images[0]
        elif model_name == "FLUX.1-dev":
            steps = 50
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
            self._apply_offload(pipe, vram_needed_gb=24)
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(generator_seed),
                callback_on_step_end=self._make_progress_callback(steps),
            ).images[0]
        elif model_name == "FLUX.1-schnell":
            steps = 4
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
            self._apply_offload(pipe, vram_needed_gb=24)
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=0.0,
                num_inference_steps=steps,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(generator_seed),
                callback_on_step_end=self._make_progress_callback(steps),
            ).images[0]
        elif model_name == "FLUX.1-dev ControlNet Depth":
            if not self.ref_image_path:
                messagebox.showinfo("Message", "Load a reference image first.")
                return
            steps = 50
            pipe = FluxControlPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16
            )
            self._apply_offload(pipe, vram_needed_gb=24)
            control_image = load_image(self.ref_image_path).resize((width, height))
            image = pipe(
                prompt,
                control_image=control_image,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=10.0,
                generator=torch.Generator("cpu").manual_seed(generator_seed),
                callback_on_step_end=self._make_progress_callback(steps),
            ).images[0]
        elif model_name == "FLUX.1-dev IP-Adapter":
            if not self.ref_image_path:
                messagebox.showinfo("Message", "Load a reference image first.")
                return
            steps = 50
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
            pipe.load_ip_adapter(
                "XLabs-AI/flux-ip-adapter",
                weight_name="ip_adapter.safetensors",
                image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
            )
            pipe.set_ip_adapter_scale(0.6)
            self._apply_offload(pipe, vram_needed_gb=24)
            ref_image = load_image(self.ref_image_path)
            image = pipe(
                prompt,
                ip_adapter_image=ref_image,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=3.5,
                generator=torch.Generator("cpu").manual_seed(generator_seed),
                callback_on_step_end=self._make_progress_callback(steps),
            ).images[0]
        elif model_name == "FLUX.1-Redux":
            if not self.ref_image_path:
                messagebox.showinfo("Message", "Load a reference image first.")
                return
            from diffusers import FluxPriorReduxPipeline
            prior_pipe = FluxPriorReduxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
            )
            self._apply_offload(prior_pipe, vram_needed_gb=4)
            ref_image = load_image(self.ref_image_path)
            pipe_prior_output = prior_pipe(ref_image)
            del prior_pipe
            torch.cuda.empty_cache()
            steps = 50
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
            )
            self._apply_offload(pipe, vram_needed_gb=24)
            image = pipe(
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=2.5,
                generator=torch.Generator("cpu").manual_seed(generator_seed),
                callback_on_step_end=self._make_progress_callback(steps),
                **pipe_prior_output,
            ).images[0]
        elif model_name == "Kolors":
            steps = 25
            pipe = KolorsPipeline.from_pretrained("Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16")
            self._apply_offload(pipe, vram_needed_gb=12)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
            image = pipe(
                prompt,
                negative_prompt="",
                guidance_scale=6.5,
                num_inference_steps=steps,
                height=height,
                width=width,
                callback_on_step_end=self._make_progress_callback(steps),
            ).images[0]
        elif model_name == "Playground-v2.5":
            steps = 50
            pipe = DiffusionPipeline.from_pretrained(
                "playgroundai/playground-v2.5-1024px-aesthetic",
                torch_dtype=torch.float16,
                variant="fp16",
            )
            self._apply_offload(pipe, vram_needed_gb=12)
            pipe.scheduler = EDMDPMSolverMultistepScheduler()
            image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=3,
                         height=height, width=width,
                         callback_on_step_end=self._make_progress_callback(steps)).images[0]
        else:
            print("Specify a supported model.\n")
            return

        # Use to validate inputs and outputs
        if DEBUG:
            print(prompt)
            print(model_name)

        self.progress_var.set(0)
        self.update_canvas_image(image)
        del pipe
        torch.cuda.empty_cache()

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
        )
        pipe.load_lora_weights("DiffusionLight/DiffusionLight", weight_name="pytorch_lora_weights.safetensors", adapter_name="diffusionlight")
        pipe.set_adapters("diffusionlight", adapter_weights=0.75)
        self._apply_offload(pipe, vram_needed_gb=12)
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
        ball_size = 256
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

    def load_reference(self):
        res = filedialog.askopenfile(initialdir="./history")
        if res:
            self.ref_image_path = res.name
            img = Image.open(self.ref_image_path)
            img.thumbnail((self.ref_canvas.winfo_width() or 400, 200), Image.LANCZOS)
            self.ref_canvas_image = ImageTk.PhotoImage(img)
            self.ref_canvas.config(height=img.height)
            self.ref_canvas.create_image(0, 0, image=self.ref_canvas_image, anchor=NW)
