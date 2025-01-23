import bpy
import os
import mathutils
import math

base_save_dir = '/mnt/merged_nvme/lht/3D/ShapeNet_renders'
base_data_dir = '/mnt/merged_nvme/lht/3D/ShapeNet_ori/'
if not os.path.exists(base_save_dir):
    os.mkdir(base_save_dir)

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 16

bpy.context.scene.cycles.max_bounces = 4  # 默认是 12，降低到 4 或更低
bpy.context.scene.cycles.diffuse_bounces = 2
bpy.context.scene.cycles.glossy_bounces = 2
bpy.context.scene.cycles.transmission_bounces = 2
bpy.context.scene.cycles.volume_bounces = 2

# 设置为 GPU 渲染
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # 可选 'CUDA', 'OPTIX', 或 'HIP'
bpy.context.scene.cycles.device = 'GPU'

# 获取所有 GPU 设备
bpy.context.preferences.addons['cycles'].preferences.get_devices()
devices = bpy.context.preferences.addons['cycles'].preferences.devices

# 启用第 2-7 个 GPU（索引从 0 开始）
for i, device in enumerate(devices):
    if device.type in {'CUDA', 'OPTIX'}:  # 确保是 GPU
        if 2 <= i <= 7 or 11 <= i <= 16:  # 仅启用 2-7 的 GPU
            device.use = True
            print(f"Enabling GPU: {device.name}: {device.type} (Index: {i})")
        else:
            device.use = False
            print(f"Disabling GPU: {device.name}: {device.type} (Index: {i})")
    else:
        device.use = False  # 禁用非 GPU 设备

print("GPU Rendering setup complete.")


bpy.context.scene.view_settings.view_transform = 'Standard'

camera = bpy.data.objects['Camera']
# 设置相机位置和朝向
camera.location = (4, -4, 3.1)
# 设置目标点为原点
target_position = mathutils.Vector((0, 0, 0))

# 计算方向并设置相机旋转
direction = target_position - camera.location
rotation_quaternion = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rotation_quaternion.to_euler()

# 设置为正交相机
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 3.0

# 设置分辨率
scene = bpy.context.scene
scene.render.resolution_x = 512
scene.render.resolution_y = 512

# 设置相机裁剪范围
camera.data.clip_start = 0.1
camera.data.clip_end = 10.0

bpy.context.scene.render.film_transparent = True

# 设置环境光
world = bpy.context.scene.world
# world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs[1].default_value = 1.0
bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1)  # white 背景
bpy.context.scene.render.film_transparent = True

# 启用节点系统
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
nodes = tree.nodes
links = tree.links

# 清空所有节点
for node in nodes:
    nodes.remove(node)

# 添加 Render Layers 节点
render_layers = nodes.new(type='CompositorNodeRLayers')

# 添加 Composite 节点
composite = nodes.new(type='CompositorNodeComposite')

# 添加 Alpha Over 节点
alpha_over = nodes.new(type='CompositorNodeAlphaOver')

# 添加纯白背景节点
white_color = nodes.new(type='CompositorNodeRGB')
white_color.outputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # 设置为纯白色

# 链接节点
links.new(render_layers.outputs['Image'], alpha_over.inputs[2])  # 渲染图像 -> Alpha Over 的第二输入
links.new(white_color.outputs['RGBA'], alpha_over.inputs[1])  # 白色背景 -> Alpha Over 的第一输入
links.new(alpha_over.outputs['Image'], composite.inputs['Image'])  # Alpha Over 输出 -> Composite 输入

def center_objects_to_bbox_center(objs, target_min=-1.0, target_max=1.0):
    """
    将物体的几何中心移动到包围盒中心。
    :param objs: 需要调整的物体列表。
    """
    # 计算所有物体的全局边界框
    all_coords = [
        mathutils.Vector(corner) @ obj.matrix_world
        for obj in objs
        for corner in obj.bound_box
    ]

    # 找到最小和最大坐标
    min_coord = mathutils.Vector((
        min(c.x for c in all_coords),
        min(-c.y for c in all_coords),
        min(-c.z for c in all_coords)
    ))
    max_coord = mathutils.Vector((
        max(c.x for c in all_coords),
        max(-c.y for c in all_coords),
        max(-c.z for c in all_coords)
    ))

    # 计算包围盒中心
    bbox_center = (min_coord + max_coord) * 0.5
    max_dim = max(max_coord - min_coord)
    scale_factor = (target_max - target_min) / max_dim

    # 将所有物体的中心平移到包围盒中心
    for obj in objs:
        obj.location -= bbox_center
        obj.scale *= scale_factor
        obj.location *= scale_factor

def handle_one_obj(file_name, obj_path, model_save_path, target_min=-1.0, target_max=1.0):
    bpy.ops.wm.obj_import(filepath=obj_path)

    for obj in bpy.data.objects:
        # 如果对象类型是ARMATURE，则删除
        if obj.type in {'ARMATURE'}:
            # print(f"Deleting object: {obj.name} of type {obj.type}")
            bpy.data.objects.remove(obj, do_unlink=True)

    # ------------------------
    # 获取所有导入的物体
    imported_objects = [obj for obj in bpy.context.scene.objects if obj.select_get()]
    mesh_objects = [obj for obj in imported_objects if obj.type == 'MESH']
    # print(f"Imported {len(mesh_objects)} objects.")
    root_object = imported_objects[0]
    # print(mesh_objects)

    center_objects_to_bbox_center(mesh_objects)

    save_file_path = os.path.join(model_save_path, file_name + ".jpg")

    bpy.context.scene.render.filepath = os.path.join(save_file_path)
    bpy.context.scene.render.image_settings.file_format = 'JPEG'

    # render
    bpy.ops.render.render(write_still=True)

    # 遍历场景中的所有对象
    for obj in bpy.data.objects:
        # print(obj)
        # 如果对象类型是 'CAMERA' 则跳过
        if obj.type == 'CAMERA':
            continue
        # 删除非 Camera 对象
        bpy.data.objects.remove(obj, do_unlink=True)

def traverse_models(base_dir):
    """
    遍历 base_dir 中的每个模型类别，提取每个模型中的 `model_normalized.obj` 文件路径和模型名称。

    Args:
        base_dir (str): 模型类别的根目录。

    Returns:
        list: 包含每个模型路径和名称的字典列表，格式为 [{"name": model_name, "obj_path": obj_path}, ...]
    """
    model_data = []

    # 遍历类别目录
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        save_category_path = os.path.join(base_save_dir, category)

        if not os.path.exists(save_category_path):
            os.mkdir(save_category_path)

        # 确保是目录
        if not os.path.isdir(category_path):
            continue

        # 遍历每个模型
        for model in os.listdir(category_path):
            model_path = os.path.join(category_path, model)
            # 确保是目录
            if not os.path.isdir(model_path):
                continue

            # 构造 `models/model_normalized.obj` 路径
            obj_path = os.path.join(model_path, "models", "model_normalized.obj")

            # 检查 .obj 文件是否存在
            if os.path.exists(obj_path):
                handle_one_obj(model, obj_path, save_category_path)


# handle_one_obj("8bc53bae41a8105f5c7506815f553527","D:/aigc3d/data_examples/shapenet/mesh/02773838/8bc53bae41a8105f5c7506815f553527/models/model_normalized.obj", "D:/aigc3d/data_examples/shapenet/objaverse_renders")

traverse_models(base_data_dir)


#
#
# import bpy
# import os
# import mathutils
# import multiprocessing
#
# # 基本路径
# base_save_dir = '/mnt/merged_nvme/lht/3D/ShapeNet_renders'
# base_data_dir = '/mnt/merged_nvme/lht/3D/ShapeNet_ori/'
# if not os.path.exists(base_save_dir):
#     os.mkdir(base_save_dir)
#
# # 只启用 GPU 2-7
# enabled_gpus = list(range(2, 8))
#
# def render_scene_with_gpu(gpu_id, models):
#     """
#     使用指定的 GPU 渲染模型场景。
#     Args:
#         gpu_id (int): 使用的 GPU ID。
#         models (list): 当前分配给该 GPU 的模型列表。
#     """
#     # 设置环境变量，限制当前进程只使用指定的 GPU
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#
#     # 初始化 Blender
#     bpy.context.scene.render.engine = 'CYCLES'
#     bpy.context.scene.cycles.device = 'GPU'
#     bpy.context.scene.cycles.samples = 8
#     bpy.context.scene.cycles.max_bounces = 4
#
#     # 设置渲染参数
#     camera = bpy.data.objects['Camera']
#     camera.location = (4, -4, 3.1)
#     camera.data.type = 'ORTHO'
#     camera.data.ortho_scale = 3.0
#
#     bpy.context.scene.render.resolution_x = 64
#     bpy.context.scene.render.resolution_y = 64
#
#     # 渲染模型
#     for model in models:
#         category, model_id, obj_path = model
#         save_path = os.path.join(base_save_dir, category, model_id + ".jpg")
#
#         # 创建目录
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#         # 导入模型
#         bpy.ops.wm.obj_import(filepath=obj_path)
#         objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
#
#         # 将物体居中并归一化
#         center_objects_to_bbox_center(objs)
#         print(f"GPU {gpu_id} 正在渲染场景 {model}....")
#         # 设置渲染保存路径
#         bpy.context.scene.render.filepath = save_path
#         bpy.ops.render.render()
#
#         # 清理场景
#         for obj in bpy.context.scene.objects:
#             if obj.type != 'CAMERA':
#                 bpy.data.objects.remove(obj, do_unlink=True)
#
# def center_objects_to_bbox_center(objs):
#     """
#     将物体的几何中心移动到包围盒中心。
#     """
#     all_coords = [
#         mathutils.Vector(corner) @ obj.matrix_world
#         for obj in objs
#         for corner in obj.bound_box
#     ]
#     min_coord = mathutils.Vector((min(c.x for c in all_coords), min(c.y for c in all_coords), min(c.z for c in all_coords)))
#     max_coord = mathutils.Vector((max(c.x for c in all_coords), max(c.y for c in all_coords), max(c.z for c in all_coords)))
#     bbox_center = (min_coord + max_coord) * 0.5
#     for obj in objs:
#         obj.location -= bbox_center
#
# def distribute_models_to_gpus(base_dir, enabled_gpus):
#     """
#     分配模型到多个 GPU 渲染任务。
#     Args:
#         base_dir (str): 模型根目录。
#         enabled_gpus (list): 启用的 GPU 列表。
#     """
#     all_models = []
#     for category in os.listdir(base_dir):
#         category_path = os.path.join(base_dir, category)
#         if not os.path.isdir(category_path):
#             continue
#
#         for model_id in os.listdir(category_path):
#             model_path = os.path.join(category_path, model_id)
#             if not os.path.isdir(model_path):
#                 continue
#
#             obj_path = os.path.join(model_path, "models", "model_normalized.obj")
#             if os.path.exists(obj_path):
#                 all_models.append((category, model_id, obj_path))
#     # print(all_models)
#     # 按 GPU 分配模型
#     num_gpus = len(enabled_gpus)
#     models_per_gpu = [[] for _ in range(num_gpus)]
#
#     for i, model in enumerate(all_models):
#         models_per_gpu[i % num_gpus].append(model)
#         # print(f"GPU {i % num_gpus} 负责: {model}")
#
#     # 启动多进程渲染
#     processes = []
#     for idx, gpu_id in enumerate(enabled_gpus):
#         print(f"GPU {gpu_id} 正在执行")
#         p = multiprocessing.Process(target=render_scene_with_gpu, args=(gpu_id, models_per_gpu[idx]))
#         processes.append(p)
#         p.start()
#
#     for p in processes:
#         p.join()
#
# if __name__ == "__main__":
#     distribute_models_to_gpus(base_data_dir, enabled_gpus)

