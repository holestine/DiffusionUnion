from tkinter import *
from tkinter import filedialog, ttk
from idlelib.tooltip import Hovertip
import threading
from transformers import SamModel, SamProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
from rembg import remove as rembg_remove
import torch
import time
from diffusers.utils import load_image, make_image_grid
import numpy as np
from PIL import Image, ImageTk
from controls import create_toolbar_button, create_number_control, load_display_image

DEBUG = False


def build_dim_composite(raw_image_path, masks, colors, dim_alpha):
    """Precompute base image + all masks at dim opacity. Called once after SAM runs."""
    base = np.array(Image.open(raw_image_path).convert("RGBA"), dtype=np.float32)
    result = base.copy()
    for mask, color in zip(masks, colors):
        src_rgb = (color * 255).astype(np.float32)
        src_a = mask.astype(np.float32) * dim_alpha
        dst_a = result[:, :, 3] / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        m = mask.astype(bool)
        out_a_m = out_a[m]
        result[m, :3] = (
            src_rgb * src_a[m, np.newaxis]
            + result[m, :3] * dst_a[m, np.newaxis] * (1.0 - src_a[m, np.newaxis])
        ) / np.maximum(out_a_m[:, np.newaxis], 1e-6)
        result[m, 3] = out_a_m * 255.0
    return Image.fromarray(result.astype(np.uint8), 'RGBA')


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

        # Segmentation state
        self.segment_masks = []      # list of bool numpy arrays (H, W)
        self.mask_colors = []        # list of (r, g, b) float arrays in [0,1]
        self.selected_masks = set()  # indices of selected masks
        self.base_image_path = None  # image path before segmentation overlay
        self.orig_image_size = None  # (w, h) of base image, cached to avoid file I/O on clicks
        self.overlay_cache = None    # last rendered PIL overlay, reused on tab switch
        self.dim_composite = None    # precomputed PIL RGBA: base + all masks at dim opacity
        self.bright_overlay_cache = {}  # per-mask bright PIL RGBA overlays, computed lazily

        # Reference to the inpainting tab (set externally after both tabs are created)
        self.inpainting_tab = None

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
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def initialize_prompts(self, parent):
        # Create text box for entering the prompt
        prompt = ""
        Label(parent, text="Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.prompt = Text(parent, height=1, wrap=WORD, pady=4)
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

        # Create textbox for entering the transparency of the segmentation
        self.transparency_entry = create_number_control(toolbar, 0.6, "Transparency", 'Enter a value from 0 to 1. Higher values generally result in higher quality images but take longer.', increment=.05, type=float, min=0, max=1)

        # Create a button to segment the image
        self.segment_button = create_toolbar_button(toolbar, 'Segment', self.segment, 'Segment the image')

        # Create a button to caption the image
        self.caption_button = create_toolbar_button(toolbar, 'Caption Image', self.caption, 'Caption the image')

        # Create a button to remove the background
        self.remove_bg_button = create_toolbar_button(toolbar, 'Remove Background', self.remove_background, 'Remove the image background using BiRefNet')

        # Indeterminate progress bar for non-diffusion operations (SAM, BLIP, rembg)
        self.progress_bar = ttk.Progressbar(toolbar, mode='indeterminate', length=200)
        self.progress_bar.pack(side=LEFT, padx=4)

    def refresh_ui(self):
        current = self.history[-1] if self.history else None
        if self.overlay_cache is not None and current == self.base_image_path:
            self.display_pil_on_canvas(self.overlay_cache)
        else:
            self._reset_segmentation()
            if current:
                self.width, self.height = load_display_image(self.canvas, current)
            else:
                self.canvas.delete("all")

        self.update_controls()

    def _reset_segmentation(self):
        self.segment_masks = []
        self.mask_colors = []
        self.selected_masks = set()
        self.base_image_path = None
        self.orig_image_size = None
        self.overlay_cache = None
        self.dim_composite = None
        self.bright_overlay_cache = {}
        if self.inpainting_tab is not None:
            self.inpainting_tab.set_segmentation_mask(None)

    def update_controls(self):
        self.prompt["state"] = DISABLED
        self.prompt['bg']    = '#D3D3D3'

    def update_canvas_image(self, image):
        self.history.append('history/{}.png'.format(time.time()))
        image.save(self.history[-1])
        self.refresh_ui()

    def display_pil_on_canvas(self, image):
        """Display a PIL image on the canvas without saving to history."""
        screen_w = self.canvas.winfo_screenwidth()
        screen_h = self.canvas.winfo_screenheight()
        img = image.copy()
        img.thumbnail((int(screen_w * 0.8), int(screen_h * 0.8)), Image.LANCZOS)
        self.canvas._display_image = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0, 0, image=self.canvas._display_image, anchor=NW)

    def on_canvas_click(self, event):
        if not self.segment_masks or not self.base_image_path:
            return

        # Map canvas coordinates to original image coordinates using cached size
        display_w = self.canvas.winfo_width()
        display_h = self.canvas.winfo_height()
        orig_w, orig_h = self.orig_image_size
        scale_x = orig_w / display_w
        scale_y = orig_h / display_h
        ix = int(event.x * scale_x)
        iy = int(event.y * scale_y)

        # Find which masks contain the clicked pixel
        hit = []
        for i, mask in enumerate(self.segment_masks):
            mask_np = np.array(mask).squeeze()
            if 0 <= iy < mask_np.shape[0] and 0 <= ix < mask_np.shape[1]:
                if mask_np[iy, ix]:
                    hit.append(i)

        if not hit:
            return

        # Toggle: if all hit masks are already selected, deselect them; otherwise select all
        if all(i in self.selected_masks for i in hit):
            for i in hit:
                self.selected_masks.discard(i)
        else:
            for i in hit:
                self.selected_masks.add(i)

        self._update_inpainting_mask()
        self.canvas.after(0, self._redraw_segmentation)

    def _redraw_segmentation(self):
        if self.dim_composite is None:
            return
        result = self.dim_composite.copy()
        for i in self.selected_masks:
            result.paste(self._get_bright_overlay(i), (0, 0), self._get_bright_overlay(i))
        self.overlay_cache = result
        self.display_pil_on_canvas(result)

    def _get_bright_overlay(self, i):
        """Return (and cache) a PIL RGBA image for mask i at full (bright) opacity."""
        if i not in self.bright_overlay_cache:
            transparency = float(self.transparency_entry.get())
            mask = self.segment_masks[i]
            color = self.mask_colors[i]
            h, w = mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[mask, :3] = (color * 255).astype(np.uint8)
            rgba[mask, 3] = int(transparency * 255)
            self.bright_overlay_cache[i] = Image.fromarray(rgba, 'RGBA')
        return self.bright_overlay_cache[i]

    def _update_inpainting_mask(self):
        """Build the combined binary mask from selected segments and push to inpainting tab."""
        if self.inpainting_tab is None or not self.base_image_path:
            return
        orig = Image.open(self.base_image_path)
        combined = np.zeros((orig.height, orig.width), dtype=np.uint8)
        for i in self.selected_masks:
            mask_np = np.array(self.segment_masks[i]).squeeze()
            combined = np.logical_or(combined, mask_np).astype(np.uint8)
        mask_image = Image.fromarray(combined * 255, mode='L')
        self.inpainting_tab.set_segmentation_mask(mask_image)

    def segment(self):
        if DEBUG:
            self.segment_thread()
        else:
            threading.Thread(target=self.segment_thread).start()

    def segment_thread(self):
        # Get all necessary arguments from UI
        prompt     = self.prompt.get('1.0', 'end-1 chars')
        model_name = self.checkpoint.get()
        transparency = float(self.transparency_entry.get())

        if len(self.history) > 0:
            init_image = load_image(self.history[-1])
            self.base_image_path = self.history[-1]
            self.orig_image_size = init_image.size  # (w, h), cached for click coordinate mapping
        else:
            init_image = Image.fromarray(np.zeros((self.width, self.height, 3), 'uint8'))
            self.base_image_path = None
            self.orig_image_size = (self.width, self.height)

        self.progress_bar.after(0, lambda: self.progress_bar.start(20))
        generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
        outputs = generator(init_image, points_per_batch=64)
        self.progress_bar.after(0, self.progress_bar.stop)

        masks = outputs["masks"]

        # Store masks with stable random colors; clear all cached render state
        self.segment_masks = [np.array(m).squeeze() for m in masks]
        self.mask_colors = [np.random.random(3) for _ in masks]
        self.selected_masks = set()
        self.overlay_cache = None
        self.bright_overlay_cache = {}

        # Clear any stale inpainting mask from a previous segmentation
        if self.inpainting_tab is not None:
            self.inpainting_tab.set_segmentation_mask(None)

        # Precompute dim composite once so per-click redraws are cheap
        self.dim_composite = build_dim_composite(
            self.base_image_path, self.segment_masks,
            self.mask_colors, transparency * 0.25
        )
        self.canvas.after(0, lambda img=self.dim_composite.copy(): self.display_pil_on_canvas(img))

        # Use to validate inputs and outputs
        if DEBUG:
            print(prompt)
            print(model_name)

    def remove_background(self):
        if DEBUG:
            self.remove_background_thread()
        else:
            threading.Thread(target=self.remove_background_thread).start()

    def remove_background_thread(self):
        self.progress_bar.after(0, lambda: self.progress_bar.start(20))
        input_image = Image.open(self.history[-1]).convert("RGBA")
        output_image = rembg_remove(input_image)
        self.progress_bar.after(0, self.progress_bar.stop)
        self.update_canvas_image(output_image)

    def remove(self):
        if DEBUG:
            self.caption()
        else:
            threading.Thread(target=self.caption).start()

    def caption(self):
        self.progress_bar.start(20)
        model_name = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        inputs = processor(images=self.history[-1], return_tensors="pt")
        out = model.generate(**inputs)
        text = [{"generated_text": processor.decode(out[0], skip_special_tokens=True)}]
        self.progress_bar.stop()

        self.prompt["state"] = NORMAL
        self.prompt.insert(END, '\n'+text[0]['generated_text'])
        self.prompt["state"] = DISABLED
