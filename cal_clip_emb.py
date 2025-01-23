import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPImageProcessor


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


def compute_and_save_embedding(preprocess, model, image_path, device, save_extension='.pt'):
    """
    计算单张图像的嵌入并保存。

    Args:
        model (clip.model.CLIP): CLIP模型。
        preprocess (callable): CLIP的预处理函数。
        image_path (str): 图像文件路径。
        device (torch.device): 设备（CPU或GPU）。
        save_extension (str): 保存嵌入的文件扩展名。默认为'.pt'。

    Returns:
        bool: 是否成功保存嵌入。
    """
    try:
        image = Image.open(image_path).convert('RGB')
        input = preprocess(images=image, return_tensors="pt", do_rescale=False).to(device)

        # image_input = preprocess(image).unsqueeze(0).to(device)  # [1, 3, H, W]
        with torch.no_grad():
            # embedding = model.encode_image(image_input)  # [1, embedding_dim]
            # embedding = embedding.cpu()  # 移动到CPU以便保存
            embedding = model.get_image_features(**input)
            embedding = embedding.cpu()
        # 构建保存路径，使用相同的基名但不同的扩展名
        base_name = os.path.splitext(image_path)[0]
        embedding_path = base_name + save_extension
        torch.save(embedding, embedding_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="使用CLIP计算目录下所有图片的图像编码器嵌入并保存到同一个文件夹中，嵌入文件名与原图一致。")
    parser.add_argument('--image_dir', type=str, default='/mnt/merged_nvme/lht/3D/ShapeNet_renders/', help='包含图像的目标目录。')
    parser.add_argument('--device', type=str, default='cuda', help="设备类型（'cuda'或'cpu'）。默认为'cuda'。")
    parser.add_argument('--save_extension', type=str, default='.pt', help="保存嵌入的文件扩展名。默认为'.pt'。")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    preprocess = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)


    # 获取所有图像文件
    image_files = get_image_files(args.image_dir)
    print(f"Found {len(image_files)} images in {args.image_dir}")

    # 遍历图像并计算保存嵌入
    success_count = 0
    fail_count = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        # 构建嵌入文件路径
        base_name = os.path.splitext(image_path)[0]
        embedding_path = base_name + args.save_extension

        # 如果嵌入文件已存在，跳过处理
        if os.path.exists(embedding_path):
            print(f"Embedding already exists for {image_path}, skipping.")
            continue

        success = compute_and_save_embedding(preprocess, model, image_path, device, args.save_extension)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"Completed. Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
