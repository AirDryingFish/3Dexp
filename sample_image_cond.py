import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
import argparse
import math

import numpy as np

import mcubes

import torch
from torchvision import transforms

import trimesh

import models_class_cond, models_ae

from pathlib import Path
from PIL import Image
from transformers import AutoModel
from transformers import CLIPModel, CLIPImageProcessor

if __name__ == "__main__":

    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae', type=str, required=True)  # 'kl_d512_m512_l16'
    parser.add_argument('--ae-pth', type=str, required=True)  # 'output/ae/kl_d512_m512_l16/checkpoint-199.pth'
    parser.add_argument('--dm', type=str, required=True)  # 'kl_d512_m512_l16_edm'
    parser.add_argument('--dm-pth', type=str,
                        required=True)  # 'output/uncond_dm/kl_d512_m512_l16_edm/checkpoint-999.pth'
    args = parser.parse_args()
    print(args)

    Path("image_cond_obj/{}".format(args.dm)).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0')

    # preprocess = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    embmodel = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant').to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ae = models_ae.__dict__[args.ae]()
    ae.eval()
    ae.load_state_dict(torch.load(args.ae_pth)['model'])
    ae.to(device)

    model = models_class_cond.__dict__[args.dm]()
    model.eval()

    model.load_state_dict(torch.load(args.dm_pth)['model'], strict=False)
    model.to(device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density + 1)  # [128]^3的网格，每个坐标上有129个点
    y = np.linspace(-1, 1, density + 1)
    z = np.linspace(-1, 1, density + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    # [3, 129, 129, 129] -> [3, 129 * 129 * 129] -> [129 * 129 * 129, 3] -> [1, 129 * 129 * 129, 3]
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device,
                                                                                                            non_blocking=True)

    total = 40
    iters = 20

    image = Image.open("/mnt/merged_nvme/lht/3D/ShapeNet_renders/04090263/dc29aaf86b072eaf1c4b0f7f86417492.jpg")
    # image = Image.open("/home/yangzengzhi/code/TRELLIS/shoes/1.jpg")
    # transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0).to(device)

    # input = preprocess(images=image, return_tensors="pt", do_rescale=False).to(device)
    # embedding = clipmodel.get_image_features(**input)

    # input = preprocess(images=image, return_tensors="pt", do_rescale=False).to(device)
    with torch.no_grad():
        embedding = embmodel(image).last_hidden_state


    print(embedding.shape)
    embedding = embedding.repeat(iters, 1, 1)
    print(embedding.shape)

    # clip_emb = torch.load("/mnt/merged_nvme/lht/3D/ShapeNet_renders/03790512/")

    with torch.no_grad():
        # print(category_id)
        for i in range(100 // iters):  # i [0, 1, ..., 9]
            print(f"current i = {i}")
            # category_id: [0, 100, 200, ..., 900]
            # batch_seeds: [0 ... 100]  [100 ... 200] ... [800 ... 900]
            sampled_array = model.sample(cond=embedding.to(device),
                                         batch_seeds=torch.arange(i * iters, (i + 1) * iters).to(device)).float()

            print(sampled_array.shape, sampled_array.max(), sampled_array.min(), sampled_array.mean(),
                  sampled_array.std())

            for j in range(sampled_array.shape[0]):
                # sampled_array 对应VAE编码的shape表面点，直接通过EMP从高斯噪声中Sample
                # grid 对应空间的均匀网格。训练中从物体内部与物体近表面中查询出物体表面，相当于插值出物体表面(某个点的occ情况)，sample时相当于直接从多个不同的网格点中插值出物体表面
                logits = ae.decode(sampled_array[j:j + 1], grid)  # j为一个Batch内的第j个物体, logits: [1, N(2048)]

                logits = logits.detach()

                volume = logits.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()
                verts, faces = mcubes.marching_cubes(volume, 0)

                verts *= gap
                verts -= 1

                m = trimesh.Trimesh(verts, faces)
                m.export('image_cond_obj/{}/gun-{:05d}.obj'.format(args.dm, i * iters + j))