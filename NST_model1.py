import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from IPython.display import display
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import wandb

wandb.login()

# Hyerparameters
total_steps = 500
img_savepoint = 100
learning_rate = 0.001
content_weight = 1
style_weight = 0.01
path = "generated.png"
imsize = 356

run = wandb.init(
	project = "NST1",
	config = {
		"total_steps": 500,
		"learning_rate": 0.001,
		"content_weight": 1,
		"style_weight": 0.01,
        "imsize": 356
	},
)

def save_checkpoint(step, model_name, optimizer):
    ckpt = {'step': step, 'optimizer_state': optimizer.state_dict()}
    torch.save(ckpt, f"{model_name}_ckpt_{str(step)}.pth")

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

content_img = load_image('lab.jpg')
style_img = load_image('painting.jpg')

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features

generated = content_img.clone().requires_grad_(True)
model = VGG().to(device).eval()

optimizer = optim.Adam([generated],lr = learning_rate)

for step in range(total_steps):
    print(step)
    # Obtain the convolution features in specifically chosen layers
    generated_features = model(generated)
    content_img_features = model(content_img)
    style_features = model(style_img)

    # Loss is 0 initially
    style_loss = content_loss = 0

    # iterate through all the features for the chosen layers
    for gen_feature, cont_feature, style_feature in zip(
        generated_features, content_img_features, style_features
    ):

        # batch_size will just be 1
        batch_size, channel, height, width = gen_feature.shape
        content_loss += torch.mean((gen_feature - cont_feature) ** 2)
        # Compute Gram Matrix of generated
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )
        # Compute Gram Matrix of Style
        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )
        style_loss += torch.mean((G - A) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % img_savepoint == 0:
        print(f"Style loss: {style_loss}")
        print(f"Content loss: {content_loss}")
        print(f"Total loss:{total_loss}")
        wandb.log({"Style loss": style_loss, "Content loss": content_loss, "Total loss": total_loss})
        wandb.log({"generated": [wandb.Image(generated, caption=f"NST image, step {step}")]})
        save_image(generated, path)
        save_checkpoint(step, "vgg-19-1", optimizer)

display(Image.open(path))

