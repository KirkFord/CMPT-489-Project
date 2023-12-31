# CMPT 489 Project - Image Recolorization for Use in Creative Domains

## Contents
Supplementary scripts: scripts I used to test color extraction, create datasets, train my main model, or create visualizers for certain models. Also includes all recorded TensorBoard data from my hyperparameter experiments.

Use PaletteNet: contains an adversarial and non-adversarial user implementation of the models i trained in pytorch. Instructions on how to use below.

Use Photo Recoloring: contains everything needed to run my modified version of the photo recoloring algorithm. Instructions on how to use below.


## Installation
Main model: if you wish to run the .py implementation of my modified PaletteNet model, please run the following in your terminal/anaconda distribution
```
pip install opencv-python
pip install matplotlib
pip install scikit-image
pip install torch
pip install torchvision
pip install tqdm
```
The baseline model is completely contained and requires no extra installation.

## Usage
Main Model: Open a file explorer and navigate to the "Use Palettenet" folder. Open a terminal in the "Use Palettenet" folder. Drop any .jpg you like into the folder and rename it "image.jpg". Run the command "python PaletteNetUse.py" for the ablated model with no adversarial training (looks better) or run "python PaletteNetUseAdv.py" to try the model with adversarial training.
You will be greeted by 4 color pickers in succession, each color you pick represents a colour in your 4-color palette. The new file will then save as "pict256.png" (or "pict256ADV.png") in "Use PaletteNet" after all of the pyplot figures are closed.

Baseline Model: Open a file explorer and navigate to the "Use Photo Recoloring" folder. Open the "Index.html" file with a web browser such as chrome, and follow the on screen instructions to add a photo. Clicking the circular coloured buttons at the bottom will allow you to change colors. When finished, click confirm and then wait for the image to generate.

## Authors and acknowledgment
The SSIM script I used was made by Nathancy, found here: https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images
Thank you for the public post.

This project would not have been possible without the use/accessibility of the following models/datasets:

My main model is a modified version of Palettenet: https://github.com/yongzx/PaletteNet-PyTorch

My baseline is a slightly modified version of palette-based Photo Recoloring: https://github.com/b-z/photo_recoloring

My main dataset is Artbench: https://github.com/liaopeiyuan/artbench
