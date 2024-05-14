from tkinter import *
from tkinter import filedialog
from diffusers.utils import load_image, make_image_grid
from diffusers import StableVideoDiffusionPipeline, AnimateDiffPipeline, DDIMScheduler, MotionAdapter, DiffusionPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import threading
from diffusers import AutoPipelineForInpainting
import torch
import os, time
import numpy as np
#from tkVideoPlayer import TkinterVideo

class image_to_vid:
    
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
        left_frame.pack(side=LEFT, fill=BOTH, expand=TRUE)

        # Create right frame
        right_frame = Frame(parent, width=self.width, height=self.height, bg='grey')
        right_frame.pack(side=RIGHT, fill=BOTH, expand=TRUE)

        return toolbar, left_frame, right_frame

    def initialize_canvas(self, left_frame):
        #self.videoplayer = TkinterVideo(master=left_frame, scaled=True)
        #self.videoplayer.pack(expand=True, fill="both")
        #self.videoplayer.load(r"generated.mp4")
        #self.videoplayer.play() # play the video

        return
        # Create a black image for the background
        self.history.append('history/{}.png'.format(time.time()))
        Image.fromarray(np.zeros((self.width,self.height,3), 'uint8')).save(self.history[-1])

        self.canvas_bg = PhotoImage(file=self.history[-1])
        self.width, self.height = self.canvas_bg.width(), self.canvas_bg.height()
        
        # Create canvas for drawing mask
        self.canvas = Canvas(left_frame, bg="black", width=self.width, height=self.height)
        self.canvas.pack(fill=BOTH, expand=False)
        self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

    def initialize_prompts(self, right_frame):
        # Create text box for entering the prompt
        self.prompt_label = Label(right_frame, text="Positive Prompt:", anchor=W)
        self.prompt_label.pack(side=TOP, fill=X, expand=FALSE)
        
        self.prompt = Text(right_frame, height=3, wrap=WORD)
        self.prompt.insert(END, "a scary monster that looks like the grim reaper dancing by a river in the mountains, high resolution, highly detailed, 8k, in a style similar to Tim Burton, realistic level of detail, pixar")
        self.prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

        # Create text box for entering negative prompt
        self.negative_prompt_label = Label(right_frame, text="Negative Prompt:", anchor=W)
        self.negative_prompt_label.pack(side=TOP, fill=X, expand=FALSE)
        
        self.negative_prompt = Text(right_frame, height=3, wrap=WORD)
        self.negative_prompt.insert(END, "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms")
        self.negative_prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

    def initialize_toolbar(self, toolbar):
        
        # Create combo box for selecting diffusion model
        model_frame = Frame(toolbar, bg='grey')
        model_frame.pack(side=LEFT, fill=X, expand=FALSE)

        self.model_label = Label(model_frame, text="Diffusion Model:", anchor=W)
        self.model_label.pack(side=LEFT, fill=Y, expand=FALSE)

        self.model = StringVar(model_frame, "ModelScopeT2V")

        model_options = ["ModelScopeT2V"]
        model_drop = OptionMenu(model_frame, self.model, *model_options)
        model_drop.pack(side=LEFT, fill=X, expand=TRUE)

        # Create button to load a background
        self.load_button = Button(toolbar, text="Load Background", command=self.load_background)
        self.load_button.pack(side=LEFT, fill=X, expand=FALSE)

        # Create button to generate the image
        self.clear_button = Button(toolbar, text="Generate Video", command=self.generate)
        self.clear_button.pack(side=LEFT, fill=X, expand=FALSE)

    def load_background(self):
        res = filedialog.askopenfile(initialdir="./history")
        if res:
            self.canvas.delete("all")
            self.history.append(res.name)
            self.canvas_bg = PhotoImage(file=self.history[-1])
            self.width, self.height = self.canvas_bg.width(), self.canvas_bg.height()
            self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)
    
    def generate(self):
        threading.Thread(target=self.generate_thread).start()

    def generate_thread(self):

        model_name = self.model.get()

        if model_name == "ModelScopeT2V":

            pipeline = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_slicing()

            prompt = self.prompt.get('1.0', 'end-1 chars')
            video_frames = pipeline(prompt).frames[0]
            export_to_video(video_frames, "generated.mp4", fps=5)


            
            self.videoplayer.load(r"generated.mp4")
            self.videoplayer.play() # play the video


        return

        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

        pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(
            "emilianJR/epiCRealism",
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipeline.scheduler = scheduler
        pipeline.enable_vae_slicing()
        pipeline.enable_model_cpu_offload()

        output = pipeline(
            prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
            negative_prompt="bad quality, worse quality, low resolution",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=torch.Generator("cpu").manual_seed(49),
        )
        frames = output.frames[0]
        export_to_video(frames, "generated.mp4", fps=7)


        return

        pipeline = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipeline.enable_model_cpu_offload()

        image = load_image("./history/small.png")
        image = image.resize((64, 64))

        generator = torch.manual_seed(42)
        frames = pipeline(image, decode_chunk_size=4, generator=generator).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)


        return

      

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

