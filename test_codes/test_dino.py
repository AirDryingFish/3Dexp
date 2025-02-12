import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
from transformers import AutoModel
from PIL import Image
from torchvision import transforms
image = Image.open("/mnt/merged_nvme/lht/3D/ShapeNet_renders/02773838/151d033af5381fd4c1e1e64c8ea306ea.jpg").resize((518, 518))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.ToTensor()
])
image = transform(image).unsqueeze(0)
model = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant')
grid_size = 518 // 14
outputs = model(image)
last_hidden_states = outputs.last_hidden_state
B, HW, C = last_hidden_states.shape
# num_register_tokens: 4 + cls token: 1
feat = last_hidden_states[:, 4 + 1:].permute(0, 2, 1).reshape(B, C, grid_size, grid_size)


