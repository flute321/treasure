import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torchvision
from PIL import Image
import torch.nn.functional as F
from banana import Network

model = Network()
path = "myFirstModel.pth"
model.load_state_dict(torch.load(path))
model.eval()

def predict(image):
    transform_crossval = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

    classes = ('sigatoka', 'pestalotiopsis', 'healthy', 'cordana')
    path = "./myFirstModel.pth"

    img = Image.open(image)
    img_t = transform_crossval(img)
    image_s= torch.unsqueeze(img_t,0)
    image_s = image_s.to(device="cpu")
    

    out = model.forward(image_s)
    conf, prediction = torch.max(out, 1)
     
    print(prediction, ' at confidence score:{0:.2f}'.format(conf))

predict('./banana/test/sigatoka/IMG_20210312_174148_1.jpg')
predict('./banana/test/sigatoka/1615455263940.jpg')

