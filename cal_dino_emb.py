# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, "
# import argparse
# import torch
# from PIL import Image
# from tqdm import tqdm
# from torchvision import transforms
# from transformers import CLIPModel, CLIPImageProcessor, AutoModel
#
#
# def get_image_files(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
#     """
#     获取指定目录下所有符合扩展名的图像文件路径。
#
#     Args:
#         directory (str): 目标目录路径。
#         extensions (list): 支持的图像文件扩展名列表。
#
#     Returns:
#         list: 图像文件的完整路径列表。
#     """
#     image_files = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if any(file.lower().endswith(ext) for ext in extensions):
#                 image_files.append(os.path.join(root, file))
#     return image_files
#
#
# def compute_and_save_embedding(preprocess, model, image_path, device, save_extension='_dino.pt'):
#     """
#     计算单张图像的嵌入并保存。
#
#     Args:
#         model (clip.model.CLIP): CLIP模型。
#         preprocess (callable): CLIP的预处理函数。
#         image_path (str): 图像文件路径。
#         device (torch.device): 设备（CPU或GPU）。
#         save_extension (str): 保存嵌入的文件扩展名。默认为'.pt'。
#
#     Returns:
#         bool: 是否成功保存嵌入。
#     """
#     try:
#         image = Image.open(image_path).convert('RGB')
#         input = preprocess(image).unsqueeze(0).to(device)
#
#         # image_input = preprocess(image).unsqueeze(0).to(device)  # [1, 3, H, W]
#         with torch.no_grad():
#             # embedding = model.encode_image(image_input)  # [1, embedding_dim]
#             # embedding = embedding.cpu()  # 移动到CPU以便保存
#             embedding = model(input)
#             embedding = embedding.cpu()
#         # 构建保存路径，使用相同的基名但不同的扩展名
#         base_name = os.path.splitext(image_path)[0]
#         embedding_path = base_name + save_extension
#         torch.save(embedding, embedding_path)
#         return True
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return False
#
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="使用CLIP计算目录下所有图片的图像编码器嵌入并保存到同一个文件夹中，嵌入文件名与原图一致。")
#     parser.add_argument('--image_dir', type=str, default='/mnt/merged_nvme/lht/3D/ShapeNet_renders/', help='包含图像的目标目录。')
#     parser.add_argument('--device', type=str, default='cuda', help="设备类型（'cuda'或'cpu'）。默认为'cuda'。")
#     parser.add_argument('--save_extension', type=str, default='_dino.pt', help="保存嵌入的文件扩展名。默认为'.pt'。")
#     args = parser.parse_args()
#
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
#
#     # preprocess = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         # transforms.ToTensor()
#     ])
#     model = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant').to(args.device)
#
#
#     # 获取所有图像文件
#     image_files = get_image_files(args.image_dir)
#     print(f"Found {len(image_files)} images in {args.image_dir}")
#
#     # 遍历图像并计算保存嵌入
#     success_count = 0
#     fail_count = 0
#     for image_path in tqdm(image_files, desc="Processing images"):
#         # 构建嵌入文件路径
#         base_name = os.path.splitext(image_path)[0]
#         embedding_path = base_name + args.save_extension
#
#         # 如果嵌入文件已存在，跳过处理
#         if os.path.exists(embedding_path):
#             print(f"Embedding already exists for {image_path}, skipping.")
#             continue
#
#         success = compute_and_save_embedding(transform, model, image_path, device, args.save_extension)
#         if success:
#             success_count += 1
#         else:
#             fail_count += 1
#
#     print(f"Completed. Success: {success_count}, Failed: {fail_count}")
#
#
# if __name__ == "__main__":
#     main()


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1, "
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel


class ImageDataset(Dataset):
    """自定义数据集类，用于批量处理图像。"""

    def __init__(self, image_files, transform):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image_path, image


def collate_fn(batch):
    # 保持 batch 的原始格式，不进行额外处理
    return batch

def compute_and_save_embeddings_batch(image_paths, embeddings, save_extension):
    """
    保存一批图像的嵌入到对应的文件中。

    Args:
        image_paths (list[str]): 图像路径列表。
        embeddings (torch.Tensor): 图像嵌入，形状为 [batch_size, embedding_dim]。
        save_extension (str): 保存嵌入文件的扩展名。
    """
    for path, embedding in zip(image_paths, embeddings):
        base_name = os.path.splitext(path)[0]
        embedding_path = base_name + save_extension
        torch.save(embedding.cpu(), embedding_path)


def get_image_files(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
    """
    获取指定目录下所有符合扩展名的图像文件路径。

    Args:
        directory (str): 目标目录路径。
        extensions (list): 支持的图像文件扩展名列表。

    Returns:
        list: 图像文件的完整路径列表。
    """
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    return image_files


def main():
    parser = argparse.ArgumentParser(
        description="使用DINOv2计算目录下所有图片的嵌入并保存，支持批量处理。")
    parser.add_argument('--image_dir', type=str, default='/mnt/merged_nvme/lht/3D/ShapeNet_renders/',
                        help='包含图像的目标目录。')
    parser.add_argument('--device', type=str, default='cuda', help="设备类型（'cuda'或'cpu'）。默认为'cuda'。")
    parser.add_argument('--save_extension', type=str, default='_dino.pt', help="保存嵌入的文件扩展名。默认为'.pt'。")
    parser.add_argument('--batch_size', type=int, default=128, help="批量大小。默认为16。")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant').to(device)
    model.eval()

    # 获取所有图像文件
    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} images in {args.image_dir}")

    # 过滤掉已存在嵌入文件的图像
    image_files = [f for f in image_files if not os.path.exists(os.path.splitext(f)[0] + args.save_extension)]
    print(f"{len(image_files)} images need processing.")

    # 创建数据集和数据加载器
    dataset = ImageDataset(image_files, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 批量处理图像
    success_count = 0
    fail_count = 0
    for batch in tqdm(dataloader, desc="Processing images"):
        image_paths, images = zip(*batch)
        images = torch.stack(images).to(device)  # 将一批图像堆叠为一个张量
        try:
            with torch.no_grad():
                embeddings = model(images).last_hidden_state # 根据你的模型输出调整这里
            compute_and_save_embeddings_batch(image_paths, embeddings, args.save_extension)
            success_count += len(image_paths)
        except Exception as e:
            print(f"Error processing batch: {e}")
            fail_count += len(image_paths)

    print(f"Completed. Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
