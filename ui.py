from tkinter import *
from PIL import Image, ImageDraw
import torch
#torch.backends.cudnn.enabled = False
import time, os
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import matplotlib.pyplot as plt

class DiffusionUnionUI:
    def __init__(self, root, background=None):
        self.root = root
        self.background_path = background
        self.root.title("Diffusion Union")

        if background:
            self.canvas_bg = PhotoImage(file=background)
            self.bg_png = load_image(background)
            self.width, self.height = self.canvas_bg.width(), self.canvas_bg.height()
        else:
            self.width, self.height = 512, 512

        # Create a left and right frame
        left_frame = Frame(root, width=self.width, height=self.height, bg='grey')
        left_frame.pack(side=LEFT, fill=BOTH, expand=False)

        right_frame = Frame(root, width=self.width, height=self.height, bg='grey')
        right_frame.pack(side=RIGHT, fill=BOTH, expand=TRUE)
        
        # Create canvas for drawing mask
        self.canvas = Canvas(left_frame, bg="black", width=self.width, height=self.height)
        self.canvas.pack(fill=BOTH, expand=True)

        if self.background_path:
            self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

        # Create a greyscale image the same size as the canvas to track mask creation
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)

        # Create text box for entering the prompt
        self.prompt_label = Label(right_frame, text="Positive Prompt:", anchor=W)
        self.prompt_label.pack(side=TOP, fill=X, expand=FALSE)
        
        self.prompt = Text(right_frame, height=3, wrap=WORD)
        self.prompt.insert(END, "a clandestine elven city with wizards doing magic, mysterious, highly detailed, 8k, sci-fi")
        self.prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

        # Create text box for entering negative prompt
        self.negative_prompt_label = Label(right_frame, text="Negative Prompt:", anchor=W)
        self.negative_prompt_label.pack(side=TOP, fill=X, expand=FALSE)
        
        self.negative_prompt = Text(right_frame, height=3, wrap=WORD)
        self.negative_prompt.insert(END, "bad anatomy, deformed, ugly")
        self.negative_prompt.pack(side=TOP, fill=BOTH, expand=TRUE)

        # Create text box for entering thickness
        thickness_frame = Frame(right_frame, bg='grey')
        thickness_frame.pack(side=TOP, fill=X, expand=FALSE)

        self.thickness_label = Label(thickness_frame, text="Thickness:", anchor=W)
        self.thickness_label.pack(side=LEFT, fill=Y, expand=FALSE)

        self.thickness = IntVar()
        self.thickness.set('2')
        options = [1,2,3,4,5,10,15,20]
        drop = OptionMenu(thickness_frame, self.thickness, *options)
        drop.pack(side=LEFT, fill=X, expand=TRUE)      

        ## Create button to clear canvas
        self.clear_button = Button(right_frame, text="Generate Image", command=self.generate)
        self.clear_button.pack(side=TOP, fill=X, expand=FALSE)

        self.generate_button = Button(right_frame, text="Clear canvas", command=self.clear_canvas)
        self.generate_button.pack(side=TOP, fill=X, expand=FALSE)

        # Add events for left mouse button on canvas. See https://python-course.eu/tkinter/events-and-binds-in-tkinter.php
        self.canvas.bind("<Button-1>",        self.start_drawing)
        self.canvas.bind("<B1-Motion>",       self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            if self.last_x and self.last_y:
                color = 'white'
                thickness = self.thickness.get()
                self.canvas.create_line(
                    self.last_x, self.last_y, x, y, fill=color, width=thickness
                )
                self.mask_editor.line(
                    [(self.last_x, self.last_y), (x, y)], fill="white", width=thickness
                )

            self.last_x = x
            self.last_y = y

    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        if self.background_path:
            self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

        self.mask.save("mask.png")


    def generate(self):   
        pipeline = AutoPipelineForInpainting.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        )
        pipeline.enable_model_cpu_offload()

        # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
        #pipeline.enable_xformers_memory_efficient_attention()

        init_image = load_image(self.background_path)
        mask_image = self.mask

        #plt.imshow(init_image)
        #plt.show()
        #plt.imshow(mask_image)
        #plt.show()

        prompt = self.prompt.get('1.0', 'end-1 chars') 
        negative_prompt = self.negative_prompt.get('1.0', 'end-1 chars') 
        image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image).images[0]
        x = make_image_grid([init_image, mask_image, image], rows=1, cols=3)

        if not os.path.exists('generated'):
            os.makedirs('generated')

        self.background_path = 'generated/{}.png'.format(time.time())
        image.save(self.background_path)
        self.canvas_bg = PhotoImage(file=self.background_path)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.canvas_bg, anchor=NW)

        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)

        


if __name__ == "__main__":
    root = Tk()
    app = DiffusionUnionUI(root, "background.png")
    root.mainloop()

