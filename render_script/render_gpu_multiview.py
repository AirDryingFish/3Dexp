import bpy
import os
import sys
import json
import mathutils
import argparse
import random
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Render models with a specific GPU.')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use for rendering')
    parser.add_argument('--model_list', type=str, required=True, help='Path to JSON file with models to render')
    parser.add_argument('--base_save_dir', type=str, required=True, help='Base directory to save renders')
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

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

def setup_cycles_gpu(gpu_id):
    # 确保 'cycles' 渲染引擎被启用
    bpy.context.scene.render.engine = 'CYCLES'

    # 获取 Cycles 的首选项
    cycles_pref = bpy.context.preferences.addons['cycles'].preferences

    # 启用 GPU 计算
    cycles_pref.compute_device_type = 'CUDA'  # 或者 'OPTIX'，取决于您的 GPU 支持情况

    # 清除现有的设备并重新添加
    cycles_pref.get_devices()

    # 禁用所有设备，除了指定的 GPU
    for device in cycles_pref.devices:
        device.use = False

    # 启用指定的 GPU
    if gpu_id < len(cycles_pref.devices):
        cycles_pref.devices[gpu_id].use = True
    else:
        print(f"GPU ID {gpu_id} 超出范围。")
        sys.exit(1)

    # 设置渲染设备为 GPU
    bpy.context.scene.cycles.device = 'GPU'

def add_light():
    # 检查是否有光源，如果没有则添加一个
    if not any(obj.type == 'LIGHT' for obj in bpy.data.objects):
        bpy.ops.object.light_add(type='SUN', location=(4, -4, 10))
        light = bpy.context.active_object
        light.data.energy = 5

def set_random_camera():
    camera = bpy.data.objects.get('Camera')

    r = 6
    random_rad_xy = math.radians(random.uniform(0, 360))
    random_rad_fai = math.radians(random.uniform(-35, 35))

    random_x = r * math.cos(random_rad_fai) * math.cos(random_rad_xy)
    random_y = r * math.cos(random_rad_fai) * math.sin(random_rad_xy)
    random_z = r * math.sin(random_rad_fai)

    if camera is None:
        bpy.ops.object.camera_add(location=(random_x, random_y, random_z))
        camera = bpy.context.active_object
    else:
        camera.location = (random_x, random_y, random_z)

    # 设置目标点为原点
    target_position = mathutils.Vector((0, 0, 0))

    # 计算方向并设置相机旋转
    direction = target_position - camera.location
    rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rotation_quaternion.to_euler()

    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 3.0
    bpy.context.scene.camera = camera

    # 设置相机裁剪范围
    camera.data.clip_start = 0.1
    camera.data.clip_end = 10.0

def setup_light_background():
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


def render_scene_with_gpu(gpu_id, models, base_save_dir):
    # 设置 GPU
    setup_cycles_gpu(gpu_id)

    # 设置渲染参数
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.max_bounces = 4

    # 添加光源
    setup_light_background()

    # 设置渲染分辨率
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # 渲染模型
    for model in models:
        category, model_id, obj_path = model

        # 导入模型
        bpy.ops.wm.obj_import(filepath=obj_path)
        objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

        # 将物体居中并归一化
        center_objects_to_bbox_center(objs)
        print(f"GPU {gpu_id} 正在渲染场景 {model} ....")

        bpy.context.scene.render.image_settings.file_format = 'JPEG'
        for i in range(24):
            set_random_camera()
            save_path = os.path.join(base_save_dir, category, model_id, f"{i}.jpg")
            # 创建目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 设置渲染保存路径
            bpy.context.scene.render.filepath = save_path
            bpy.ops.render.render(write_still=True)

        print(f"GPU {gpu_id} 渲染场景 {model} 完成")

        # 清理场景
        for obj in bpy.context.scene.objects:
            if obj.type != 'CAMERA' and obj.type != 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)


def main():
    args = parse_args()
    gpu_id = args.gpu_id
    model_list_path = args.model_list
    base_save_dir = args.base_save_dir

    with open(model_list_path, 'r') as f:
        models = json.load(f)

    render_scene_with_gpu(gpu_id, models, base_save_dir)


if __name__ == "__main__":
    # 启用 OBJ 导入插件
    # bpy.ops.wm.addon_enable(module="io_scene_obj")
    main()
