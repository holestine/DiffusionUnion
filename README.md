# DiffusionUnion
Adding some application logic around some diffusion models to make their capabilities a little more accessible.

# Prepare the environment
'''
conda create --name du python=3.10
conda activate du
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 accelerate xformers diffusers transformers matplotlib -c pytorch -c nvidia -c xformers -c conda-forge 
'''

# Notes
Noncontinuous mask seems to cause artifacts in other areas of the images

# Things I use for testing that will be removed
'''
conda deactivate
conda remove -n du --all

python -c "import torch;print(torch.cuda.is_available())"
python -c "import torch;torch.zeros(1).cuda()"

python -c "import torch;print(torch.__version__)"
'''

