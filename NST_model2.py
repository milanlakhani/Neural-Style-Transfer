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
alpha = 1
beta = 0.01
path = "generated.png"
imsize = 356

run = wandb.init(
	project = "NST1",
	config = {
		"total_steps": 500,
		"learning_rate": 0.001,
		"alpha": 1,
		"beta": 0.01,
        "imsize": 356
	},
)

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

content_img = load_image('lab.jpg')
style_imgs = [load_image('painting.jpg'), load_image('femme-pleure.jpg')]

generated = content_img.clone().requires_grad_(True)
model = VGG().to(device).eval()

optimizer = optim.Adam([generated],lr = learning_rate)
# optimizer = optim.LBFGS([content_img])

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
    # Obtain the convolution features in specifically chosen layers
    generated_features = model(generated)
    content_img_features = model(content_img)
    style_imgs_features = [model(style_img) for style_img in style_imgs]

    # Loss is 0 initially
    style_loss = content_loss = 0

    # iterate through all the features for the chosen layers
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

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % img_savepoint == 0:
        print(f"Style loss: {style_loss}")
        print(f"Content loss: {content_loss}")
        print(f"Total loss:{total_loss}")
        save_image(generated, path)

display(Image.open(path))

