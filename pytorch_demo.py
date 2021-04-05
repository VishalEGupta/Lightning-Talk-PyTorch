import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
img = Image.open("bevo.jpg")

resnet = models.resnet101(pretrained=True)

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
img = Image.open("bevo.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()
_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]