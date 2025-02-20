import torch
import torch.nn as nn
import torch.nn.functional as F
import extensions.FlexiCubes.examples.render as df_renderer
import extensions.FlexiCubes.examples.util as utils
from extensions.FlexiCubes.examples.loss import sdf_reg_loss
import numpy as np

class MeshLoss(nn.Module):
    def __init__(self, sdf_regularizer=1.0):
        """初始化损失类，设置正则化参数等"""
        super(MeshLoss, self).__init__()
        self.sdf_regularizer = sdf_regularizer
        self.renderer = df_renderer

    def compute_normal_loss(self, pred_normals, target_normals):
        """计算法线损失：通过均方误差（MSE）计算预测法线和目标法线之间的差异"""
        return F.mse_loss(pred_normals, target_normals)

    def compute_depth_loss(self, pred_depth, target_depth, target_mask):
        """计算深度损失：通过加权的平方误差计算深度损失"""
        depth_loss = (((pred_depth - target_depth) * target_mask) ** 2).sum(-1)
        return torch.sqrt(depth_loss + 1e-8).mean() * 10

    def compute_mask_loss(self, pred_mask, target_mask):
        """计算掩码损失：通过平均绝对误差计算掩码的损失"""
        return (pred_mask - target_mask).abs().mean()

    def compute_sdf_loss(self, pred_sdf, gt_sdf):
        """计算SDF损失：通过均方误差（MSE）计算预测SDF和真实SDF之间的差异"""
        return F.mse_loss(pred_sdf, gt_sdf)

    def compute_reg_loss(self, sdf, grid_edges, weight):
        """计算正则化损失：包括SDF正则化、L_dev正则化和权重的L1正则化"""
        # 根据t_iter逐步调整SDF正则化权重
        # sdf_weight = self.sdf_regularizer - (self.sdf_regularizer - self.sdf_regularizer / 20) * min(1.0, 4.0 * t_iter)
        reg_loss = sdf.reg_loss(sdf, grid_edges).mean()  # SDF正则化
        reg_loss += (weight[:, :20]).abs().mean() * 0.2  # 权重的L1正则化
        return reg_loss

    def get_random_cam(self, batch_size, fovy = np.deg2rad(45), iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=1.5, device="cuda"):
        def get_random_camera():
            proj_mtx = utils.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])
            mv     = utils.translate(0, 0, -cam_radius) @ utils.random_rotation_translation(0.125)
            mvp    = proj_mtx @ mv
            return mv, mvp
        mv_batch = []
        mvp_batch = []
        for i in range(batch_size):
            mv, mvp = get_random_camera()
            mv_batch.append(mv)
            mvp_batch.append(mvp)
        return torch.stack(mv_batch).to(device), torch.stack(mvp_batch).to(device)
        # return mv_batch, mvp_batch

    def forward(self, predicted_mesh_list, target_mesh_list, sdf, grid_edges, weight, L_dev, train_res = 512):
        """计算总损失"""
        batch_size = len(predicted_mesh_list)
        cam_mv, cam_mvp = self.get_random_cam(batch_size)
        mask_list_pred = []
        mask_list_target = []
        depth_list_pred = []
        depth_list_target = []
        reg_loss = []
        for i_mesh in batch_size:
            pred = self.renderer.render_mesh_paper(predicted_mesh_list[i_mesh], cam_mv[i_mesh].unsqueeze(0), cam_mvp[i_mesh].unsqueeze(0), train_res)
            target = self.renderer.render_mesh_paper(target_mesh_list[i_mesh], cam_mv[i_mesh].unsqueeze(0),
                                                   cam_mvp[i_mesh].unsqueeze(0), train_res)
            mask_list_pred.append(pred['mask'])
            mask_list_target.append(target['mask'])
            depth_list_pred.append(pred['depth'])
            depth_list_target.append(target['depth'])

            # 计算正则化损失
            reg_loss.append(self.compute_reg_loss(sdf[i_mesh], grid_edges, weight[i_mesh]))

        mask_pred = torch.stack(mask_list_pred) # [B, 512, 512, 1]
        mask_target = torch.stack(mask_list_target)
        depth_pred = torch.stack(depth_list_pred) # [B, 512, 512, 4]
        depth_target = torch.stack(depth_list_target)

        reg_loss = torch.cat(reg_loss).mean()

        total_mask_loss = self.compute_mask_loss(mask_pred, mask_target)
        total_depth_loss = self.compute_depth_loss(depth_pred, depth_target, mask_target)

        reg_loss += L_dev * 0.5  # L_dev正则化
        # 汇总所有损失
        total_loss = total_mask_loss + total_depth_loss + reg_loss

        return total_loss
