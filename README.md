# DiffusionUnion

Adding some application logic around some diffusion models to make their capabilities a little more accessible.

# Prepare the environment

```
conda create --name du python=3.10 -y
conda activate du
python -m pip install -U pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 accelerate xformers diffusers transformers matplotlib -c pytorch -c nvidia -c xformers -c conda-forge -y
pip install opencv-python
```

# Guide
I've created an application that makes some of the image editing and content creation models more accessible to those who want to work images and avoid the code complexities. You'll need a GPU to run most of these, mine is 12GB.

## Inpainting
This tab exposes the features of the Stable Diffisuion, Stable Diffusion XL and Kandinsky models for inpainting and the Latent Diffusion Model for super resolution. I've tried to make the UI intuitive by adding tooltips and enabling or disabling the relevant features as the model selection changes for example the negative prompt is only valid for the Kandinsky model. The following shows a sequence of operations that I used to generate an image from a black background, you can also load any image for a starting point. If you don't like the results of the generation there is an undo button to go back and all generated images are saved in the history folder.

![](./assets/1.png)
*This was generated with the current prompts by masking out the entire image.*

![](./assets/2.png)
*Mask out the left hand side and change the prompt add something to the scene. Also used the Kandinsky model.*

![](./assets/3.png)
*Mask out another area and modify the prompt to adjust the scene.*

![](./assets/4.png)
*This is the result of the last operation*

![](./assets/5.png)
*Use the mask to select an area of interest and the super res operation*

![](./assets/6.png)
*This is the result of that operation which increases the image size by about a factor of 4 in each direction. If it's size is less than 1024x1024 you can use the Stable Diffusion XL model to further increase it to that size*

![](./assets/7.png)
*And this is the full size 1024x1024 image from the history folder*

## Some other generated images

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![](./assets/creature.png)|![](./assets/snake.png)|![](./assets/river_cat_1.png)|

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![](./assets/ufo.png)|![](./assets/ship.png)|![](./assets/river_cat_2.png)|
