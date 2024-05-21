from tkinter import *
from tkinter import filedialog
from idlelib.tooltip import Hovertip
import threading
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import time

DEBUG = False

class image_generation_ui:

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
        prompt = "a photograph of a highly forested area in the arctic near a frozen waterfall with rocky cliffs, highly detailed, 8k, realistic"
        Label(parent, text="Positive Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.prompt = Text(parent, height=1, wrap=WORD)
        self.prompt.insert(END, prompt)
        self.prompt.pack(side=TOP, fill=BOTH, expand=True)

    def initialize_toolbar(self, toolbar):
        
        # Create combo box for selecting a diffusion model
        checkpoint_frame = Frame(toolbar, bg='grey')
        checkpoint_options = ["Stable Diffusion 2.1"]
        self.checkpoint = StringVar(checkpoint_frame, checkpoint_options[0])
        Hovertip(checkpoint_frame, 'Select the model to use')
        Label(checkpoint_frame, text="Model", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        checkpoint_menu = OptionMenu(checkpoint_frame, self.checkpoint, *checkpoint_options)
        checkpoint_menu.config(width=20)
        checkpoint_menu.pack(side=LEFT, fill=X, expand=True)
        checkpoint_frame.pack(side=LEFT, fill=X, expand=False)

        # Create a button to load an image
        self.load_button = Button(toolbar, text="Load Image", command=self.load_background)
        Hovertip(self.load_button, 'Open an image')
        self.load_button.pack(side=LEFT, fill=X, expand=False)

        # Create a button to generate the image
        self.generate_button = Button(toolbar, text="Generate Image", command=self.generate)
        Hovertip(self.generate_button, 'Generate a new image')
        self.generate_button.pack(side=LEFT, fill=X, expand=False)

        # Create a button to revert changes
        self.undo_button = Button(toolbar, text="Undo", command=self.undo)
        Hovertip(self.undo_button,'Undo the last generated image')
        self.undo_button.pack(side=LEFT, fill=X, expand=False)

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
        else:
            print("Specify a supported model.\n")
            return

        # Use to validate inputs and outputs
        if DEBUG:
            print(prompt)
            print(model_name)

        self.update_canvas_image(image)
        
    def load_background(self):
        res = filedialog.askopenfile(initialdir="./history")
        if res:
            self.history.append(res.name)
            self.refresh_ui()
            self.update_controls()
