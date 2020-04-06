from PIL import Image
import urllib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# model = models.alexnet(pretrained=True)
model = models.mnasnet1_0(pretrained=True)
model.eval()

# Download an example image from the pytorch website
url, filename = (
    "https://www.cats.org.uk/media/3236/choosing-a-cat.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    print("exception occured")
    urllib.request.urlretrieve(url, filename)


input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
print(input_tensor.shape)
# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
normalised_output = F.softmax(output[0], dim=0)

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

top_values, top_indices = torch.topk(normalised_output, 5)
for i in range(5):
    percentage = top_values[i] * 100
    index = top_indices[i]
    print("%{} - {}".format(percentage.round(), classes[index]))

# v, index = torch.max(normalised_output, 0)
# percentage = v * 100
# print("I am {}% certain that is {}.".format(percentage, classes[index]))

# _, index = torch.max(output, 1)

# percentage = torch.nn.functional.softmax(output[0], dim=0)[0] * 100
# print(percentage)
# print(classes[index[0]], percentage[index[0]].item())
