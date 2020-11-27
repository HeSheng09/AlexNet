import torch
from AlexNet import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("./data/dog01.jpeg")
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# create model
model = AlexNet(num_classes=len(classes))
# load model weights
model_weight_path = "checkpoints/AlexNet202011251637.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(classes[predict_cla], predict[predict_cla].item())
plt.title("{},{:.3f}".format(classes[predict_cla],predict[predict_cla].item()))
plt.show()
