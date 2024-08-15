!pip install wandb
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from IPython.display import display
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import wandb
import yaml

import vgg

WANDB_API_KEY = None

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)

with open("config.yaml", 'r') as file:
    settings = yaml.safe_load(file)

# Hyerparameters
total_steps = settings["total_steps"]
img_savepoint = settings["img_savepoint"]
learning_rate = settings["learning_rate"]
content_weight = settings["content_weight"]
style_weight = settings["style_weight"]
path = settings["path"]
imsize = settings["imsize"]
style_images = settings["style_images"]
content_image = settings["content_image"]

if WANDB_API_KEY:
    run = wandb.init(
        project = "NST1",
        config = {
            "total_steps": total_steps,
            "learning_rate": learning_rate,
            "content_weight": content_weight,
            "style_weight": style_weight,
            "imsize": imsize,
            "style_images": style_images,
            "content_image": content_image,
            "optimizer": "ADAM"
		},
	)

def save_checkpoint(step, model_name, optimizer, generated, path):
    ckpt = {
    'step': step,
    'optimizer_state': optimizer.state_dict(),
    }
    torch.save(ckpt, f"checkpoints/{model_name}_ckpt_step_{str(step)}.pth")
    save_image(generated, path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose(
    [
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
    ]
)

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def load_checkpoint(file_name, optimizer, generated_path=None, device=None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(file_name, map_location=device)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    print("Optimizer's state loaded!")
    if generated_path:
        generated = load_image(generated_path)
    print("Generated image loaded!")
    return generated

content_img = load_image(f"pictures/{content_image}")
style_imgs = [load_image(f"pictures/{img}") for img in style_images]

generated = content_img.clone().requires_grad_(True)
model = vgg.VGG().to(device).eval()

optimizer = optim.Adam([generated],lr = learning_rate)

def covariance_matrix(features):
    batch_size, channel, height, width = features.size()
    features = features.view(channel, height * width)
    mean = features.mean(1, keepdim=True)
    cov = (features - mean).mm((features - mean).t()) / (height * width - 1)
    return cov

def vincent_loss(generated_feature, style_feature):
    gen_cov = covariance_matrix(generated_feature)
    style_cov = covariance_matrix(style_feature)
    return torch.mean((gen_cov - style_cov) ** 2)

for step in range(total_steps):
    print(step)
    generated_features = model(generated)
    content_img_features = model(content_img)
    style_imgs_features = [model(style_img) for style_img in style_imgs]

    style_loss = content_loss = 0

    # Iterate through all the features for the chosen layers
    for gen_feature, cont_feature, style_features in zip(
        generated_features, content_img_features, list(map(list, zip(*style_imgs_features)))
    ):

        # batch_size will just be 1
        batch_size, channel, height, width = gen_feature.shape
        content_loss += torch.mean((gen_feature - cont_feature) ** 2)

        if step % 2 == 0:
            # Compute Gram Matrix of generated
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )
            # Compute Gram Matrix of Style
            for style_feature in style_features:
                A = style_feature.view(channel, height * width).mm(
                    style_feature.view(channel, height * width).t()
                )
                style_loss += torch.mean((G - A) ** 2)
        else:
            # Compute Vincent's Loss
            for style_feature in style_features:
                style_loss += vincent_loss(gen_feature, style_feature)

    style_loss = style_loss / len(style_imgs)

    total_loss = content_weight * content_loss + style_weight * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step + 1 % img_savepoint == 0:
        print(f"Style loss: {style_loss}")
        print(f"Content loss: {content_loss}")
        print(f"Total weighted loss:{total_loss}")
        if WANDB_API_KEY:
            wandb.log({"Style loss": style_loss, "Content loss": content_loss, "Total loss": total_loss})
            wandb.log({"generated": [wandb.Image(generated, caption=f"NST image, step {step}")]})
        save_checkpoint(step, "vgg-19-2", optimizer, generated, path)

display(Image.open(path))
