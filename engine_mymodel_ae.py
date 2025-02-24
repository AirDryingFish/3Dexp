# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from extensions.FlexiCubes.examples.render import get_rotate_camera
import extensions.FlexiCubes.examples.render as df_renderer
import numpy as np
import os
'''
@params:
    model: 模型
    criteron: 损失函数
    data_loader: 数据集
    optimizer: 优化器
    device: gpus
    epoch: 当前训练epoch
    loss_scaler: 防止梯度下溢
'''

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    kl_weight = 1e-3

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # 返回data_loader的一个batch
    for data_iter_step, (surface, gt_mesh, feat) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        surface = surface.to(device, non_blocking=True)
        feat = feat.to(device, non_blocking=True)
        for item in gt_mesh:
            item.vertices = item.vertices.to(device, non_blocking=True)
            item.faces = item.faces.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            # model进行inference
            outputs = model(surface, feat)  # [B, 2048]
            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]  # loss_kl : [B]
            else:
                loss_kl = None

            # outputs = outputs['list_mesh']
            # # [B, N(2048)]
            # # 选择所有的行和第0到第1024列的数据(空间中随机点体积volume数据)
            # loss_vol = criterion(outputs[:, :1024], labels[:, :1024])
            # # 选择所有的行和第1024列到最后一列的数据(近平面near surface数据)
            # loss_near = criterion(outputs[:, 1024:], labels[:, 1024:])
            #
            #
            # if loss_kl is not None:
            #     loss = loss_vol + 0.1 * loss_near + kl_weight * loss_kl
            # else:
            #     loss = loss_vol + 0.1 * loss_near
            loss = criterion(outputs['list_mesh'], gt_mesh, outputs['sdf'], outputs['grid_edges'], outputs['weight'], outputs['L_dev'])
            loss += + kl_weight * loss_kl
        loss_value = loss.item()

        # threshold = 0
        # # 前1024列的output，代表体积的查询
        # pred = torch.zeros_like(outputs[:, :1024])
        # pred[outputs[:, :1024] >= threshold] = 1

        # accuracy = (pred == labels[:, :1024]).float().sum(dim=1) / labels[:, :1024].shape[1]
        # accuracy = accuracy.mean()
        # intersection = (pred * labels[:, :1024]).sum(dim=1)
        # union = (pred + labels[:, :1024]).gt(0).sum(dim=1) + 1e-5
        # iou = intersection * 1.0 / union
        # iou = iou.mean()

        # 当loss不合理时停止训练
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 考虑梯度累积和scaler
        loss /= accum_iter
        # scaler每隔accum_iter之后进行loss反向传播更新参数
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        # 只有当到达梯度累计iter时才清空梯度
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # 同步多gpu的数据，等待所有gpu数据计算完成之后再进行后续的操作
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        # metric_logger.update(loss_vol=loss_vol.item())
        # metric_logger.update(loss_near=loss_near.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        # metric_logger.update(iou=iou.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        # 计算所有设备上的loss均值
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, display_res=[512,512],
             print_freq=50):
    """
    修改后的 evaluate 函数：
    - 数据读取格式与 train 保持一致，即返回 (surface, gt_mesh, feat)
    - 不计算训练 loss，而是使用当前网络输出提取预测网格，并渲染生成图片进行可视化
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # 切换到评估模式
    model.eval()

    for data_iter_step, (surface, gt_mesh_batch, feat) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # 将 surface 放到 device 上
        surface = surface.to(device, non_blocking=True)
        feat = feat.to(device, non_blocking=True)
        # 对 gt_mesh_batch 中的每个 Mesh 对象，也将其数据放到 device 上
        for mesh in gt_mesh_batch:
            mesh.vertices = mesh.vertices.to(device, non_blocking=True)
            mesh.faces = mesh.faces.to(device, non_blocking=True)


        # 此处可以调用模型进行推理（如果需要），例如：
        outputs = model(surface, feat)
        # 此处我们不计算 loss，而是使用后处理模块 fc 得到预测网格
        # 假设 grid_verts、sdf、cube_fx8、weight 等在数据集中或前面已经计算好
        # 注意：这里 training=False，表示在评估时提取预测网格
        # vertices, faces, L_dev = fc(grid_verts, sdf, cube_fx8, voxel_grid_res,
        #                             beta_fx12=weight[:, :12],
        #                             alpha_fx8=weight[:, 12:20],
        #                             gamma_f=weight[:, 20],
        #                             training=False)
        flexicubes_mesh = outputs['list_mesh'][0]
        # 计算面法线（或自动计算顶点法线），用于渲染
        flexicubes_mesh.auto_normals()

        # 获取旋转相机参数（例如通过一个计数器 it//save_interval 控制不同视角）
        mv, mvp = get_rotate_camera(data_iter_step, iter_res=display_res, device=device, use_kaolin=False)
        # 注意，这里 mv、mvp 需要扩展 batch 维度为 1
        val_buffers = df_renderer.render_mesh_paper(flexicubes_mesh, mv.unsqueeze(0), mvp.unsqueeze(0),
                                               display_res, return_types=["normal"], white_bg=True)
        # 将渲染出的法线图转换为图像
        val_image = ((val_buffers["normal"][0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

        # 同时对 ground truth 网格进行渲染
        gt_buffers = df_renderer.render_mesh_paper(gt_mesh_batch[0], mv.unsqueeze(0), mvp.unsqueeze(0),
                                              display_res, return_types=["normal"], white_bg=True)
        gt_image = ((gt_buffers["normal"][0].detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

        # 拼接预测与 gt 图像，并保存
        import imageio
        combined = np.concatenate([val_image, gt_image], axis=1)
        out_dir = f"val_output_images/{epoch}"

        imageio.imwrite(os.path.join(out_dir, f'{data_iter_step}.png'), combined)
        print(f"Evaluation Step [{epoch}], saved visualization image.")

        # 同时可以更新 metric_logger 中的统计信息（如果需要）
        # metric_logger.update(dummy=0)  # 这里仅作为示例更新

    metric_logger.synchronize_between_processes()
    print("Evaluation finished.")

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}