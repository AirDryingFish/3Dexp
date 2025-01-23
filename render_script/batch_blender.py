import os
import subprocess
import json

# 基本路径
base_save_dir = '/mnt/merged_nvme/lht/3D/ShapeNet_renders_multiview_24'
base_data_dir = '/mnt/merged_nvme/lht/3D/ShapeNet_ori/'

blender_executable = '/home/yangzengzhi/Blender/blender-4.3.0-linux-x64/blender'  # 请替换为 Blender 的实际路径
render_script = '/home/yangzengzhi/code/3DShape2VecSet/render_script/render_gpu_multiview.py'  # Blender渲染的脚本
blender_setting = '/home/yangzengzhi/code/3DShape2VecSet/render_script/shapenet_setting.blend'

# 启用的 GPU 列表
enabled_gpus = [2, 3, 4, 5, 6, 7]

if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)


def distribute_models_to_gpus(base_dir, enabled_gpus):
    """
    分配模型到多个 GPU 渲染任务。
    Args:
        base_dir (str): 模型根目录。
        enabled_gpus (list): 启用的 GPU 列表。
    """
    all_models = []
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        for model_id in os.listdir(category_path):
            model_path = os.path.join(category_path, model_id)
            if not os.path.isdir(model_path):
                continue

            obj_path = os.path.join(model_path, "models", "model_normalized.obj")
            if os.path.exists(obj_path):
                all_models.append((category, model_id, obj_path))

    # 按 GPU 分配模型
    num_gpus = len(enabled_gpus)
    models_per_gpu = [[] for _ in range(num_gpus)]

    for i, model in enumerate(all_models):
        models_per_gpu[i % num_gpus].append(model)

    # 为每个 GPU 创建一个模型列表文件
    model_lists = []
    for idx, gpu_id in enumerate(enabled_gpus):
        gpu_models = models_per_gpu[idx]
        model_list_path = f'/tmp/models_gpu_{gpu_id}.json'
        with open(model_list_path, 'w') as f:
            json.dump(gpu_models, f)
        model_lists.append((gpu_id, model_list_path))

    # 启动独立的 Blender 进程
    processes = []

    for gpu_id, model_list_path in model_lists:
        cmd = [
            blender_executable,
            '-b', blender_setting,
            '--log-level', '0',
            '--python', render_script,
            '--',
            '--gpu_id', str(gpu_id),
            '--model_list', model_list_path,
            '--base_save_dir', base_save_dir,
        ]
        print(f"启动 GPU {gpu_id} 的渲染进程")
        p = subprocess.Popen(cmd)
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.wait()


if __name__ == "__main__":
    distribute_models_to_gpus(base_data_dir, enabled_gpus)
