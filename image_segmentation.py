from tkinter import *
from tkinter import filedialog, ttk
from idlelib.tooltip import Hovertip
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
from rembg import remove as rembg_remove
import torch
import numpy as np
from PIL import Image, ImageTk
from controls import TabBase, create_toolbar_button, create_number_control, load_display_image

DEBUG = False


def build_dim_composite(raw_image_path, masks, colors, dim_alpha):
    """Precompute base image + all masks composited at dim opacity using standard alpha blending.
    Called once after SAM runs so per-click redraws only need to paste selected masks on top."""
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


class image_segmentation_ui(TabBase):

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

        # Cached models — loaded once, reused across calls
        self._sam_pipeline = None
        self._blip_processor = None
        self._blip_model = None

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
        """Create and pack the toolbar, left canvas frame, and right prompt frame."""
        toolbar = Frame(parent, width=2*self.width, height=20, bg='light grey')
        toolbar.pack(side=TOP, fill=X, expand=False)

        left_frame = Frame(parent, width=self.width, height=self.height, bg='grey')
        left_frame.pack(side=LEFT, fill=BOTH, expand=False)

        right_frame = Frame(parent, width=self.width, height=self.height, bg='grey')
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        return toolbar, left_frame, right_frame

    def initialize_canvas(self, parent):
        """Create the image canvas and bind left-click for segment selection."""
        self.canvas = Canvas(parent, bg="black", width=self.width, height=self.height)
        self.canvas.pack(fill=BOTH, expand=False)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def initialize_prompts(self, parent):
        """Create the caption output text box (read-only; populated by BLIP)."""
        Label(parent, text="Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.prompt = Text(parent, height=1, wrap=WORD, pady=4)
        self.prompt.pack(side=TOP, fill=BOTH, expand=True)

    def initialize_toolbar(self, toolbar):
        """Build the toolbar: model selector, transparency control, and action buttons."""
        # Model selector (currently only SAM is supported)
        checkpoint_frame = Frame(toolbar, bg='grey')
        checkpoint_options = ["Segment Anything"]
        self.checkpoint = StringVar(checkpoint_frame, checkpoint_options[0])
        Hovertip(checkpoint_frame, 'Select the model to use')
        Label(checkpoint_frame, text="Model", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        checkpoint_menu = OptionMenu(checkpoint_frame, self.checkpoint, *checkpoint_options)
        checkpoint_menu.config(width=20)
        checkpoint_menu.pack(side=LEFT, fill=X, expand=True)
        checkpoint_frame.pack(side=LEFT, fill=X, expand=False)

        # Controls mask overlay opacity (0 = invisible, 1 = opaque)
        self.transparency_entry = create_number_control(toolbar, 0.6, "Transparency",
            'Enter a value from 0 to 1.', increment=.05, type=float, min=0, max=1)

        create_toolbar_button(toolbar, 'Segment', self.segment, 'Segment the image')
        create_toolbar_button(toolbar, 'Caption Image', self.caption, 'Caption the image')
        create_toolbar_button(toolbar, 'Remove Background', self.remove_background,
            'Remove the image background using BiRefNet')

        # Indeterminate progress bar for non-diffusion operations (SAM, BLIP, rembg)
        self.progress_bar = ttk.Progressbar(toolbar, mode='indeterminate', length=200)
        self.progress_bar.pack(side=LEFT, padx=4)

    def refresh_ui(self):
        """Refresh the canvas. If the history image changed since segmentation, reset state."""
        current = self.history[-1] if self.history else None
        if self.overlay_cache is not None and current == self.base_image_path:
            # Show cached overlay — no need to re-render
            self.display_pil_on_canvas(self.overlay_cache)
        else:
            # New image in history; clear segmentation and show the current image
            self._reset_segmentation()
            if current:
                self.width, self.height = load_display_image(self.canvas, current)
            else:
                self.canvas.delete("all")
        self.update_controls()

    def _reset_segmentation(self):
        """Clear all segmentation state and notify the inpainting tab to clear its mask."""
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
        """Keep the prompt box read-only (it's populated programmatically by BLIP)."""
        self.prompt["state"] = DISABLED
        self.prompt['bg']    = '#D3D3D3'

    def display_pil_on_canvas(self, image):
        """Scale a PIL image to 80% of screen and display it on the canvas without saving to history."""
        screen_w = self.canvas.winfo_screenwidth()
        screen_h = self.canvas.winfo_screenheight()
        img = image.copy()
        img.thumbnail((int(screen_w * 0.8), int(screen_h * 0.8)), Image.LANCZOS)
        self.canvas._display_image = ImageTk.PhotoImage(img)  # keep reference to prevent GC
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.create_image(0, 0, image=self.canvas._display_image, anchor=NW)

    def on_canvas_click(self, event):
        """Toggle the segment(s) under the clicked pixel in/out of the selection."""
        if not self.segment_masks or not self.base_image_path:
            return

        # Map canvas coordinates → original image coordinates using the cached scale
        display_w = self.canvas.winfo_width()
        display_h = self.canvas.winfo_height()
        orig_w, orig_h = self.orig_image_size
        ix = int(event.x * orig_w / display_w)
        iy = int(event.y * orig_h / display_h)

        # Find every mask that covers the clicked pixel
        hit = [
            i for i, mask in enumerate(self.segment_masks)
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]
        ]
        if not hit:
            return

        # If all hit masks are already selected, deselect them; otherwise select all
        if all(i in self.selected_masks for i in hit):
            self.selected_masks.difference_update(hit)
        else:
            self.selected_masks.update(hit)

        self._update_inpainting_mask()
        self.canvas.after(0, self._redraw_segmentation)

    def _redraw_segmentation(self):
        """Redraw the overlay: dim composite base + bright overlays for selected masks."""
        if self.dim_composite is None:
            return
        result = self.dim_composite.copy()
        for i in self.selected_masks:
            overlay = self._get_bright_overlay(i)
            result.paste(overlay, (0, 0), overlay)
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
        """OR all selected segment masks together and push the result to the inpainting tab."""
        if self.inpainting_tab is None or not self.base_image_path:
            return
        orig = Image.open(self.base_image_path)
        combined = np.zeros((orig.height, orig.width), dtype=np.uint8)
        for i in self.selected_masks:
            combined = np.logical_or(combined, self.segment_masks[i]).astype(np.uint8)
        self.inpainting_tab.set_segmentation_mask(Image.fromarray(combined * 255, mode='L'))

    # ------------------------------------------------------------------ segment

    def segment(self):
        """Run SAM segmentation in a background thread."""
        self.run(self.segment_thread)

    def segment_thread(self):
        """Load (or reuse) the SAM pipeline, run mask generation, and display the result."""
        transparency = float(self.transparency_entry.get())

        if self.history:
            from diffusers.utils import load_image
            init_image = load_image(self.history[-1])
            self.base_image_path = self.history[-1]
            self.orig_image_size = init_image.size  # cache for click coordinate mapping
        else:
            init_image = Image.fromarray(np.zeros((self.height, self.width, 3), 'uint8'))
            self.base_image_path = None
            self.orig_image_size = (self.width, self.height)

        self.progress_bar.after(0, lambda: self.progress_bar.start(20))

        # Load the SAM pipeline once and reuse it on subsequent calls
        if self._sam_pipeline is None:
            self._sam_pipeline = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)
        outputs = self._sam_pipeline(init_image, points_per_batch=64)

        self.progress_bar.after(0, self.progress_bar.stop)

        # Store masks with stable random colors and clear all prior render state
        self.segment_masks = [np.array(m).squeeze() for m in outputs["masks"]]
        self.mask_colors = [np.random.random(3) for _ in self.segment_masks]
        self.selected_masks = set()
        self.overlay_cache = None
        self.bright_overlay_cache = {}

        # Clear any stale mask in the inpainting tab from a previous segmentation
        if self.inpainting_tab is not None:
            self.inpainting_tab.set_segmentation_mask(None)

        # Precompute the dim composite once; subsequent clicks only paste selected bright overlays
        self.dim_composite = build_dim_composite(
            self.base_image_path, self.segment_masks, self.mask_colors, transparency * 0.25
        )
        self.canvas.after(0, lambda img=self.dim_composite.copy(): self.display_pil_on_canvas(img))

    # ------------------------------------------------------------------ caption

    def caption(self):
        """Run BLIP image captioning in a background thread."""
        self.run(self.caption_thread)

    def caption_thread(self):
        """Load (or reuse) BLIP, generate a caption, and append it to the prompt box."""
        self.progress_bar.after(0, lambda: self.progress_bar.start(20))

        # Load the BLIP processor and model once and reuse them on subsequent calls
        if self._blip_processor is None:
            self._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self._blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        inputs = self._blip_processor(images=Image.open(self.history[-1]), return_tensors="pt")
        out = self._blip_model.generate(**inputs)
        text = self._blip_processor.decode(out[0], skip_special_tokens=True)

        self.progress_bar.after(0, self.progress_bar.stop)

        # UI updates must happen on the main thread
        def update_ui():
            self.prompt["state"] = NORMAL
            self.prompt.insert(END, '\n' + text)
            self.prompt["state"] = DISABLED
        self.prompt.after(0, update_ui)

    # -------------------------------------------------------- remove background

    def remove_background(self):
        """Remove the image background in a background thread using BiRefNet via rembg."""
        self.run(self.remove_background_thread)

    def remove_background_thread(self):
        """Run rembg background removal and save the result to history."""
        self.progress_bar.after(0, lambda: self.progress_bar.start(20))
        output_image = rembg_remove(Image.open(self.history[-1]).convert("RGBA"))
        self.progress_bar.after(0, self.progress_bar.stop)
        # update_canvas_image must be called on the main thread
        self.canvas.after(0, lambda: self.update_canvas_image(output_image))
