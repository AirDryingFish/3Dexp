import bpy

# 启用 Cycles 渲染引擎
bpy.context.scene.render.engine = 'CYCLES'

# 设置为 GPU 渲染
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # 可选 'CUDA', 'OPTIX', 或 'HIP'
bpy.context.scene.cycles.device = 'GPU'

# 启用特定 GPU 设备
bpy.context.preferences.addons['cycles'].preferences.get_devices()
devices = bpy.context.preferences.addons['cycles'].preferences.devices
# print(devices)
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

# print("GPU Rendering setup complete.")