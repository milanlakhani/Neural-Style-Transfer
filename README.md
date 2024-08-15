## Introduction

This project is about using Neural Style Transfer to make provided portraits or landscapes take the style of art. The repository contains a simple model, `baseline.py` and two more complex models, `complex1.py` and `complex2.py`. The more complex models are capable of taking any number of style images and combining these styles to the content image. One of these models also combines together two different Style Loss functions, while the other employs the LBFGS optimizer.

Since the model itself does not need training for Neural Style Transfer, no database was needed. You can find pictures, and add your own too, in the `pictures` directory.


## Setup

The Python version used to make this was Python 3.12.3
To install the requirements for this project, after cloning navigate to the root directory on the command line, and run `pip install -r requirements.txt`.

You can download a checkpoint file, `vgg-19-c2_ckpt_step_finished.pth`, from this link, https://drive.google.com/file/d/1L1eR9Lg0elt4udTtVZqzxTS6SOPWoNjD/view?usp=drive_link
You should add the checkpoint to the root directory.


## Training

You can set parameter values, including which images are used, for all models in `config.yaml`.
These include:
 - total_steps (The total number of updates to the image that the model makes)
 - img_savepoint (The frequency at which images and metrics are saved)
 - learning_rate (The learning rate when using ADAM optimizer)
 - content_weight (The weight attributed to content image loss in the total loss calculation)
 - style_weight (The weight attributed to style image loss in the total loss calculation)
 - path (The path where the generated image is saved)
 - imsize (The size of the image in pixels)
 - style_images (A list of the style images to be used. Note that this must remain a list, even if it only contains one image. In the case of the baseline model, the first style image in the list will be used. Do not write the full path for each, just the filename.)
 - content_image (The content image which will be used. Do not write the full path, just the filename.)
 - style_loss_type (The type of style loss function that will be used, for complex model 2)


## Wandb

If you want to record results of a model, near the top of the model file (`baseline.py` `complex1.py` or `complex2.py`), update the value of `WANDB_API_KEY` from `None` to your API key. During training you will see results in project `NST1`.

See wandb reports here: ##########TODO Wandb reports!!!!!!!!!!


## Inference

Open the root folder, which contains the inference file, and add checkpoint `vgg-19-c2_ckpt_step_finished.pth` to the folder if you have not already, as detailed in the Setup section.

Run all the cells of the inference notebook in order. If using Google Colab, you will need to first save the directory in your drive, then at the top of this notebook add
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Neural-Style-Transfer
```
assuming the last line is the correct location of the directory. In the folder `pictures` I have provided images which can be used to test the model, these can be set in config.yaml as detailed above.
