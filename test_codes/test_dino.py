import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
from torchvision import transforms


# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("/mnt/merged_nvme/lht/3D/ShapeNet_renders/02773838/151d033af5381fd4c1e1e64c8ea306ea.jpg").resize((518, 518))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.ToTensor()
])

# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-giant')
image = transform(image).unsqueeze(0)
model = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant')

# inputs = processor(images=image, return_tensors="pt")
outputs = model(image)
last_hidden_states = outputs.last_hidden_state



# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
#
# processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
# model = AutoModel.from_pretrained('facebook/dinov2-large')
#
# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state