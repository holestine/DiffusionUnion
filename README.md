# DiffusionUnion
Adding some application logic around some diffusion models to make their capabilities a little more accessible.

# Prepare the environment
'''
conda create --name du python=3.10
conda activate du
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 accelerate xformers diffusers transformers matplotlib -c pytorch -c nvidia -c xformers -c conda-forge 
pip install tkvideoplayer
'''


I've tried to make the UI intuitive by adding tooltips and enabling or disabling the relevant features as the model selection changes