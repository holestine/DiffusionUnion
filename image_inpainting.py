from tkinter import *
from tkinter import messagebox, ttk
from idlelib.tooltip import Hovertip
from diffusers.utils import load_image, make_image_grid
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import threading
from diffusers import AutoPipelineForInpainting, LDMSuperResolutionPipeline, FluxFillPipeline, StableDiffusionUpscalePipeline
import torch
import time
import warnings
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

from controls import TabBase, create_number_control, create_toolbar_button, load_display_image

DEBUG = False


class inpainting_ui(TabBase):

    def __init__(self, parent, history, width=512, height=512):

        # Efficient attention is not native in old PyTorch versions and is needed to reduce GPU usage
        self.use_efficient_attention = int(torch.__version__.split('.')[0]) < 2

        # Size of image to work with
        self.width = width
        self.height = height

        # Images created this session
        self.history = history

        # Get frames needed for layout
        toolbar, toolbar2, left_frame, right_frame = self.create_layout(parent)

        # Populate controls
        self.initialize_toolbar(toolbar, toolbar2)
        self.initialize_prompts(right_frame)
        self.initialize_canvas(left_frame)
        self.update_controls()

        # State variables
        self.drawing = False
        self.current_mask = []         # oval coordinates drawn by the user, replayed on refresh
        self.segmentation_mask = None  # binary mask pushed from the segmentation tab

    def create_layout(self, parent):
        """Create two toolbar rows, a left canvas frame, and a right prompt frame."""
        toolbar_container = Frame(parent, bg='light grey')
        toolbar_container.pack(side=TOP, fill=X, expand=False)
        toolbar = Frame(toolbar_container, bg='light grey')
        toolbar.pack(side=TOP, fill=X, expand=False)
        toolbar2 = Frame(toolbar_container, bg='light grey')
        toolbar2.pack(side=TOP, fill=X, expand=False)

        left_frame = Frame(parent, width=self.width, height=self.height, bg='grey')
        left_frame.pack(side=LEFT, fill=BOTH, expand=False)

        right_frame = Frame(parent, width=self.width, height=self.height, bg='grey')
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        return toolbar, toolbar2, left_frame, right_frame

    def initialize_canvas(self, parent):
        """Create the canvas, bind mouse events for mask painting, and create a blank mask."""
        self.canvas = Canvas(parent, bg="black", width=self.width, height=self.height)
        self.canvas.pack(fill=BOTH, expand=False)
        # See https://python-course.eu/tkinter/events-and-binds-in-tkinter.php for binding info
        self.canvas.bind("<Button-1>",        self.start_drawing)
        self.canvas.bind("<B1-Motion>",       self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Greyscale mask image; white pixels mark the region to inpaint
        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)

    def initialize_prompts(self, parent):
        """Create positive and negative prompt text boxes."""
        prompt = "a photograph of an alien space ship with a metallic and mechanical appearance hovering above the ground in a highly forested area in the arctic near a frozen waterfall and rocky cliffs, highly detailed, 8k, realistic, stars in the sky, colorful"
        Label(parent, text="Positive Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.prompt = Text(parent, height=1, wrap=WORD, pady=4)
        self.prompt.insert(END, prompt)
        self.prompt.pack(side=TOP, fill=BOTH, expand=True)

        Label(parent, text="Negative Prompt:", anchor=W).pack(side=TOP, fill=X, expand=False)
        self.negative_prompt = Text(parent, height=1, wrap=WORD, pady=4)
        self.negative_prompt.insert(END, "bad anatomy, deformed, ugly, poor details, blurry")
        self.negative_prompt.pack(side=TOP, fill=BOTH, expand=True)

    def initialize_toolbar(self, toolbar, toolbar2):
        """Build toolbar row 1 (model + generation controls) and row 2 (super res + outpaint)."""
        # Row 1: model selection and drawing controls
        checkpoint_frame = Frame(toolbar, bg='grey')
        checkpoint_options = ["Stable Diffusion 1.5", "Stable Diffusion XL 1.5", "Kandinsky 2.2", "FLUX.1 Fill"]
        self.checkpoint = StringVar(checkpoint_frame, checkpoint_options[0])
        Hovertip(checkpoint_frame, 'Select the diffusion model to use')
        Label(checkpoint_frame, text="Model", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        checkpoint_menu = OptionMenu(checkpoint_frame, self.checkpoint, *checkpoint_options)
        checkpoint_menu.config(width=20)
        checkpoint_menu.pack(side=LEFT, fill=X, expand=True)
        checkpoint_frame.pack(side=LEFT, fill=X, expand=False)
        self.checkpoint.trace_add("write", self.checkpoint_selection_callback)

        self.radius_entry          = create_number_control(toolbar, 100, 'Brush Radius',    'Radius of the brush used to draw the mask', min=1)
        self.blur_entry            = create_number_control(toolbar, 33,  'Blur Factor',     'Amount for the mask to blend with the original image.', min=0)
        self.generator_entry       = create_number_control(toolbar, 1,   'Generator',       'Different int values produce different results.', min=1)
        self.inference_steps_entry = create_number_control(toolbar, 50,  'Inference Steps', 'Higher values produce better images but take longer.', min=1, max=999)
        self.strength_entry        = create_number_control(toolbar, 0.8, 'Strength',        'How strongly the model modifies the masked area.', increment=0.05, type=float, min=0.01, max=1.0)
        create_toolbar_button(toolbar, "Generate",   self.generate,    'Generate a new image')
        create_toolbar_button(toolbar, "Clear Mask", self.clear_mask,  'Clear the current mask')
        self.undo_button = create_toolbar_button(toolbar, "Undo", self.undo, 'Undo the last generated image', RIGHT)

        # Row 2: super resolution controls
        super_res_frame = Frame(toolbar2, bg='grey')
        super_res_options = ["LDM (small images)", "SD x4 Upscaler (large images)"]
        self.super_res_model = StringVar(super_res_frame, super_res_options[0])
        Hovertip(super_res_frame, 'Select the super resolution model to use')
        Label(super_res_frame, text="Super Res Model", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        super_res_menu = OptionMenu(super_res_frame, self.super_res_model, *super_res_options)
        super_res_menu.config(width=20)
        super_res_menu.pack(side=LEFT, fill=X, expand=True)
        super_res_frame.pack(side=LEFT, fill=X, expand=False)
        create_toolbar_button(toolbar2, "Super Res", self.super_res, 'Increase the image resolution')

        # Outpainting controls (FLUX.1-Fill only)
        outpaint_dir_frame = Frame(toolbar2, bg='grey')
        outpaint_directions = ["Left", "Right", "Top", "Bottom", "All"]
        self.outpaint_direction = StringVar(outpaint_dir_frame, "Right")
        Hovertip(outpaint_dir_frame, 'Direction to expand the image')
        Label(outpaint_dir_frame, text="Outpaint Direction", anchor=W).pack(side=LEFT, fill=Y, expand=False)
        outpaint_menu = OptionMenu(outpaint_dir_frame, self.outpaint_direction, *outpaint_directions)
        outpaint_menu.config(width=8)
        outpaint_menu.pack(side=LEFT, fill=X, expand=True)
        outpaint_dir_frame.pack(side=LEFT, fill=X, expand=False)
        self.outpaint_amount = create_number_control(toolbar2, 256, 'Expand px', 'Pixels to add in the chosen direction', min=64, max=2048)
        create_toolbar_button(toolbar2, "Outpaint", self.outpaint, 'Expand the image using FLUX.1-Fill')

        # Progress bar — updated via callback_on_step_end during inference
        self.progress_var = DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(toolbar2, variable=self.progress_var, maximum=100, length=200)
        self.progress_bar.pack(side=LEFT, padx=4)

    def checkpoint_selection_callback(self, *args):
        """Re-evaluate which controls are active when the model selection changes."""
        self.update_controls()

    def _set_steps(self, value):
        """Programmatically set the inference steps entry to a model-appropriate default."""
        self.inference_steps_entry.delete(0, END)
        self.inference_steps_entry.insert(0, str(value))

    def update_controls(self):
        """Enable or disable controls based on the selected model's supported parameters."""
        checkpoint = self.checkpoint.get()
        if checkpoint in ("Stable Diffusion 1.5", "Stable Diffusion XL 1.5"):
            self.negative_prompt['state'] = DISABLED
            self.negative_prompt['bg'] = '#D3D3D3'
            self.radius_entry['state'] = NORMAL
            self.blur_entry['state'] = NORMAL
            self.generator_entry['state'] = NORMAL
            self.inference_steps_entry['state'] = NORMAL
            self.strength_entry['state'] = NORMAL
            self._set_steps(50)
        elif checkpoint == "Kandinsky 2.2":
            self.negative_prompt["state"] = NORMAL
            self.negative_prompt['bg'] = '#FFFFFF'
            self.radius_entry['state'] = NORMAL
            self.blur_entry['state'] = DISABLED
            self.generator_entry['state'] = DISABLED
            self.inference_steps_entry['state'] = DISABLED
            self.strength_entry['state'] = DISABLED
        elif checkpoint == "FLUX.1 Fill":
            self.negative_prompt['state'] = DISABLED
            self.negative_prompt['bg'] = '#D3D3D3'
            self.radius_entry['state'] = NORMAL
            self.blur_entry['state'] = NORMAL
            self.generator_entry['state'] = NORMAL
            self.inference_steps_entry['state'] = NORMAL
            self.strength_entry['state'] = DISABLED
            self._set_steps(50)

        self.undo_button["state"] = NORMAL if len(self.history) >= 1 else DISABLED

    def set_segmentation_mask(self, mask_pil):
        """Accept a binary mask from the segmentation tab and apply it as the current mask."""
        self.segmentation_mask = mask_pil
        if mask_pil is not None:
            self.mask = mask_pil.resize((self.width, self.height), Image.LANCZOS)
        else:
            self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)
        self.current_mask = []

    def _draw_mask_overlay(self):
        """Composite a white semi-transparent overlay onto the canvas wherever the mask is set."""
        mask_arr = np.array(self.mask)
        if mask_arr.max() == 0:
            return
        mask_display = self.mask.resize((self.width, self.height), Image.NEAREST)
        mask_arr = np.array(mask_display)
        overlay = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=np.uint8)
        overlay[mask_arr > 0] = [255, 255, 255, 140]
        overlay_img = Image.fromarray(overlay, 'RGBA')
        self.canvas._mask_overlay = ImageTk.PhotoImage(overlay_img)  # keep reference to prevent GC
        self.canvas.create_image(0, 0, image=self.canvas._mask_overlay, anchor=NW)

    def clear_mask(self):
        """Clear the painted mask, any segmentation mask, and refresh the canvas."""
        self.current_mask = []
        self.segmentation_mask = None
        self.mask = Image.new("L", (self.width, self.height))
        self.mask_editor = ImageDraw.Draw(self.mask)
        self.refresh_ui()

    def refresh_ui(self):
        """Reload the current history image and redraw the mask overlay on top."""
        if len(self.history) > 0:
            self.canvas.delete("all")
            new_width, new_height = load_display_image(self.canvas, self.history[-1])
            # If the image dimensions changed, resize the mask to match
            if new_width != self.width or new_height != self.height:
                self.width, self.height = new_width, new_height
                self.mask = self.mask.resize((self.width, self.height), Image.LANCZOS)
                self.mask_editor = ImageDraw.Draw(self.mask)
        else:
            self.canvas.delete("all")

        # Replay brush strokes on top of the refreshed image
        for (left, top, right, bottom) in self.current_mask:
            self.canvas.create_oval(left, top, right, bottom, fill="white", outline="")

        self._draw_mask_overlay()
        self.update_controls()

    def undo(self):
        """Remove the most recent image from history and redisplay the previous one."""
        self.history.pop()
        self.refresh_ui()

    # ------------------------------------------------------------ mask painting

    def start_drawing(self, event):
        """Begin a mask paint stroke on mouse button press."""
        self.drawing = True

    def draw(self, event):
        """Paint a circular brush stroke on both the canvas and the PIL mask image."""
        if self.drawing:
            x, y = event.x, event.y
            radius = int(self.radius_entry.get())
            left, top, right, bottom = x-radius, y-radius, x+radius, y+radius
            # Draw visible stroke on the canvas
            self.canvas.create_oval(left, top, right, bottom, fill="white", outline="")
            # Draw the same stroke on self.mask (used during inference)
            self.mask_editor.ellipse([(left, top), (right, bottom)], fill=(255))
            # Track strokes so they can be replayed on refresh
            self.current_mask.append((left, top, right, bottom))

    def stop_drawing(self, event):
        """End the current mask paint stroke on mouse button release."""
        self.drawing = False

    # -------------------------------------------------------------- super res

    def super_res(self):
        """Run the selected super resolution model in a background thread."""
        if self.super_res_model.get() == "SD x4 Upscaler (large images)":
            self.run(self.sd_x4_upscaler_thread)
        else:
            self.run(self.super_res_thread)

    def super_res_thread(self):
        """LDM 4× super resolution. Crops to the masked region before upscaling.
        Designed for small inputs (~128px); feeding large images blows up activation memory."""
        model_id = "CompVis/ldm-super-resolution-4x-openimages"
        pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self._apply_offload(pipeline, vram_needed_gb=2)

        low_res_img = Image.open(self.history[-1]).convert("RGB")

        # Crop to the masked bounding box if a mask is set
        y, x = np.nonzero(self.mask)
        if len(x) > 0 and len(y) > 0:
            low_res_img = low_res_img.crop((min(x), min(y), max(x), max(y)))

        # LDM was designed for ~128px inputs; larger inputs blow up activation memory
        max_input_size = 128
        if max(low_res_img.size) > max_input_size:
            low_res_img.thumbnail((max_input_size, max_input_size), Image.LANCZOS)

        try:
            super_res_image = pipeline(low_res_img, num_inference_steps=100, eta=1).images[0]
            self.update_canvas_image(super_res_image)
        except Exception as ex:
            print(ex)
            messagebox.showinfo("Error", ex.args[0])
        finally:
            del pipeline
            torch.cuda.empty_cache()

    def sd_x4_upscaler_thread(self):
        """SD x4 Upscaler — for larger inputs. Crops to masked region, then upscales up to 512px input → 2048px output."""
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
        )
        self._apply_offload(pipeline, vram_needed_gb=8)

        prompt = self.prompt.get('1.0', 'end-1 chars')
        low_res_img = Image.open(self.history[-1]).convert("RGB")

        # Crop to the masked bounding box if a mask is set
        y, x = np.nonzero(self.mask)
        if len(x) > 0 and len(y) > 0:
            low_res_img = low_res_img.crop((min(x), min(y), max(x), max(y)))

        # Model accepts up to 512×512 input and outputs 4× that (up to 2048×2048)
        low_res_img.thumbnail((512, 512), Image.LANCZOS)

        try:
            super_res_image = pipeline(prompt=prompt, image=low_res_img).images[0]
            self.update_canvas_image(super_res_image)
        except Exception as ex:
            print(ex)
            messagebox.showinfo("Error", ex.args[0])
        finally:
            del pipeline
            torch.cuda.empty_cache()

    # --------------------------------------------------------------- outpaint

    def outpaint(self):
        """Expand the canvas in the selected direction in a background thread."""
        self.run(self.outpaint_thread)

    def outpaint_thread(self):
        """Build an expanded canvas + directional mask and run FLUX.1-Fill outpainting."""
        prompt    = self.prompt.get('1.0', 'end-1 chars')
        direction = self.outpaint_direction.get()
        amount    = int(self.outpaint_amount.get())
        seed      = int(self.generator_entry.get())

        source = Image.open(self.history[-1]).convert("RGB")
        src_w, src_h = source.size

        # Compute new canvas size, paste offset, and mask box for the expansion direction
        if direction == "Right":
            new_w, new_h = src_w + amount, src_h
            paste_x, paste_y = 0, 0
            mask_box = (src_w, 0, new_w, new_h)
        elif direction == "Left":
            new_w, new_h = src_w + amount, src_h
            paste_x, paste_y = amount, 0
            mask_box = (0, 0, amount, new_h)
        elif direction == "Bottom":
            new_w, new_h = src_w, src_h + amount
            paste_x, paste_y = 0, 0
            mask_box = (0, src_h, new_w, new_h)
        elif direction == "Top":
            new_w, new_h = src_w, src_h + amount
            paste_x, paste_y = 0, amount
            mask_box = (0, 0, new_w, amount)
        else:  # All sides
            new_w, new_h = src_w + 2*amount, src_h + 2*amount
            paste_x, paste_y = amount, amount
            mask_box = None

        # Build the expanded image with the source pasted at the correct offset
        canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        canvas.paste(source, (paste_x, paste_y))

        # Build the outpaint mask: white = region to generate, black = original pixels
        mask = Image.new("L", (new_w, new_h), 0)
        if mask_box:
            ImageDraw.Draw(mask).rectangle(mask_box, fill=255)
        else:
            # All sides: fill everything white then punch out the original image area
            ImageDraw.Draw(mask).rectangle((0, 0, new_w, new_h), fill=255)
            ImageDraw.Draw(mask).rectangle((paste_x, paste_y, paste_x + src_w, paste_y + src_h), fill=0)

        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
        self._apply_offload(pipe, vram_needed_gb=24)
        steps = 50
        self.progress_var.set(0)
        result = pipe(
            prompt=prompt, image=canvas, mask_image=mask,
            height=new_h, width=new_w, num_inference_steps=steps,
            guidance_scale=30,
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=self._make_progress_callback(steps),
        ).images[0]

        # Paste the original pixels back to preserve them exactly
        result.paste(source, (paste_x, paste_y))

        self.progress_var.set(0)
        del pipe
        torch.cuda.empty_cache()
        self.update_canvas_image(result)

    # --------------------------------------------------------------- generate

    def generate(self):
        """Run inpainting with the selected model in a background thread."""
        self.run(self.generate_thread)

    def generate_thread(self):
        """Read UI params, dispatch to the selected model method, then save the result."""
        prompt              = self.prompt.get(         '1.0', 'end-1 chars')
        negative_prompt     = self.negative_prompt.get('1.0', 'end-1 chars')
        generator_seed      = int(self.generator_entry.get())
        blur_factor         = int(self.blur_entry.get())
        num_inference_steps = int(self.inference_steps_entry.get())
        strength            = float(self.strength_entry.get())
        model_name          = self.checkpoint.get()

        self.progress_var.set(0)

        init_image = (load_image(self.history[-1]) if self.history
                      else Image.fromarray(np.zeros((self.height, self.width, 3), 'uint8')))

        dispatch = {
            "Stable Diffusion 1.5":    self._inpaint_sd15,
            "Stable Diffusion XL 1.5": self._inpaint_sdxl,
            "Kandinsky 2.2":           self._inpaint_kandinsky,
            "FLUX.1 Fill":             self._inpaint_flux_fill,
        }
        fn = dispatch.get(model_name)
        if fn is None:
            print("Specify a supported model.")
            return

        image = fn(init_image, prompt, negative_prompt, generator_seed,
                   blur_factor, num_inference_steps, strength)

        if DEBUG:
            plt.imshow(make_image_grid([init_image, self.mask, image], rows=1, cols=3))
            plt.show()

        self.progress_var.set(0)
        self.clear_mask()
        self.update_canvas_image(image)

    def _inpaint_sd15(self, init_image, prompt, negative_prompt,
                      generator_seed, blur_factor, num_inference_steps, strength):
        """Inpaint with Stable Diffusion 1.5 (runwayml/stable-diffusion-inpainting)."""
        pipe = AutoPipelineForInpainting.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
        )
        self._apply_offload(pipe, vram_needed_gb=10)
        if self.use_efficient_attention:
            pipe.enable_xformers_memory_efficient_attention()
        mask = pipe.mask_processor.blur(self.mask.resize(init_image.size), blur_factor=blur_factor)
        image = pipe(
            prompt=prompt, image=init_image, mask_image=mask,
            generator=torch.Generator("cpu").manual_seed(generator_seed),
            strength=strength, num_inference_steps=num_inference_steps,
            callback_on_step_end=self._make_progress_callback(num_inference_steps),
        ).images[0]
        del pipe
        torch.cuda.empty_cache()
        return image

    def _inpaint_sdxl(self, init_image, prompt, negative_prompt,
                      generator_seed, blur_factor, num_inference_steps, strength):
        """Inpaint with Stable Diffusion XL (diffusers/stable-diffusion-xl-1.0-inpainting-0.1)."""
        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
        )
        self._apply_offload(pipe, vram_needed_gb=12)
        if self.use_efficient_attention:
            pipe.enable_xformers_memory_efficient_attention()
        mask = pipe.mask_processor.blur(self.mask.resize(init_image.size), blur_factor=blur_factor)
        image = pipe(
            prompt=prompt, image=init_image, mask_image=mask,
            generator=torch.Generator("cpu").manual_seed(generator_seed),
            strength=strength, num_inference_steps=num_inference_steps,
            callback_on_step_end=self._make_progress_callback(num_inference_steps),
        ).images[0]
        del pipe
        torch.cuda.empty_cache()
        return image

    def _inpaint_kandinsky(self, init_image, prompt, negative_prompt,
                           generator_seed, blur_factor, num_inference_steps, strength):
        """Inpaint with Kandinsky 2.2. Supports negative prompt; ignores blur/steps/strength."""
        pipe = AutoPipelineForInpainting.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        )
        self._apply_offload(pipe, vram_needed_gb=12)
        if self.use_efficient_attention:
            pipe.enable_xformers_memory_efficient_attention()
        image = pipe(
            prompt=prompt, negative_prompt=negative_prompt,
            image=init_image, mask_image=self.mask.resize(init_image.size),
        ).images[0]
        del pipe
        torch.cuda.empty_cache()
        return image

    def _inpaint_flux_fill(self, init_image, prompt, negative_prompt,
                           generator_seed, blur_factor, num_inference_steps, strength):
        """Inpaint with FLUX.1-Fill. Composites the original pixels back outside the mask
        to prevent the transformer from subtly degrading unmasked regions."""
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16
        )
        self._apply_offload(pipe, vram_needed_gb=24)
        mask = pipe.mask_processor.blur(self.mask.resize(init_image.size), blur_factor=blur_factor)
        flux_output = pipe(
            prompt=prompt, image=init_image, mask_image=mask,
            height=init_image.height, width=init_image.width,
            num_inference_steps=num_inference_steps, guidance_scale=30,
            generator=torch.Generator("cpu").manual_seed(generator_seed),
            callback_on_step_end=self._make_progress_callback(num_inference_steps),
        ).images[0]
        # Use FLUX output only inside the mask; keep original pixels elsewhere
        binary_mask = self.mask.resize(init_image.size).convert("L")
        image = Image.composite(flux_output, init_image, binary_mask)
        del pipe
        torch.cuda.empty_cache()
        return image

    # ---------------------------------------------------------------- helpers

    def _apply_offload(self, pipe, vram_needed_gb):
        """Move the pipeline to GPU if VRAM is sufficient, otherwise use sequential CPU offload."""
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_vram_gb >= vram_needed_gb:
            pipe.to("cuda")
        else:
            pipe.enable_sequential_cpu_offload()
            if hasattr(pipe, 'vae'):
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()

    def _make_progress_callback(self, num_steps):
        """Return a diffusers step callback that updates the progress bar each inference step."""
        def callback(_pipe, step, _timestep, kwargs):
            self.progress_var.set((step + 1) / num_steps * 100)
            return kwargs
        return callback
