from functools import wraps
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from torch.utils.checkpoint import checkpoint
import einops

from torch_cluster import fps

from timm.models.layers import DropPath
import pointops
from extensions.FlexiCubes.flexicubes import FlexiCubes
import extensions.FlexiCubes.examples.render as df_render
from extensions.FlexiCubes.examples import util
from extensions.FlexiCubes.examples.util import *

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        # return val if exists(context) else x
        # 选择使用cross attention或者self attention
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = einops.rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = einops.repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = einops.rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi  # [2pi, 4pi, 8pi, ..., 2^(8)pi]
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C
        return embed

class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean # [B, M, C0]
        self.logvar = logvar # [B, M, C0]
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            # 计算与标准正态分布之间的KL散度
            if other is None: # [B, M, C0] -> [B]
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError

class GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = (
            self.linear_q(feat),
            self.linear_k(feat),
            self.linear_v(feat),
        )
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat

class Block(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(Block, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = (
            self.attn(feat, coord, reference_index)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat, coord, reference_index)
        )
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]


class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points):
        coord, feat, offset = points
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)
        for block in self.blocks:
            points = block(points, reference_index)
        return points


class GVAPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_dim,
        embed_dim,
        groups,
        neighbors = 16,
        qkv_bias=True,
        pe_multiplier=True,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_dim
        self.embed_channels = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim, bias=False),
            PointBatchNorm(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_dim,
            groups=groups,
            neighbours=neighbors,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])


class KLAutoEncoder(nn.Module):
    def __init__(
            self,
            *,
            encoder_depth=24,
            depth=24,
            dim=512,
            queries_dim=512,

            patch_embed_groups=8,
            patch_embed_neighbours=8,

            output_dim=1,
            num_inputs=2048,
            num_latents=512,
            latent_dim=64,
            heads=8,
            dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            voxel_grid_res=64,
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.k = patch_embed_neighbours
        self.groups = patch_embed_groups


        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_pos_embed = PointEmbed(dim=queries_dim)

        self.point_embed = GVAPatchEmbed(
            in_dim=1536,
            embed_dim=512,
            groups=patch_embed_groups,
            depth=1,
            neighbors=patch_embed_neighbours,
            qkv_bias=True,
            pe_multiplier=True,
            pe_bias=True,
            attn_drop_rate=0.0,
            enable_checkpoint=False,
        )

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.encoder_layers = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(encoder_depth):
            self.encoder_layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))


        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads=1, dim_head=dim),
                                          context_dim=dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        # self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()
        self.fc_params_predictor = nn.Linear(queries_dim, 1 + 3 + 21)
        self._init_weights()


        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

        self.fc = FlexiCubes()
        self.voxel_grid_res = voxel_grid_res
        self.register_buffer("x_nx3", self.fc.construct_voxel_grid(self.voxel_grid_res)[0])
        self.register_buffer("cube_fx8", self.fc.construct_voxel_grid(self.voxel_grid_res)[1])

        self.all_edges = self.cube_fx8[:, self.fc.cube_edges].reshape(-1, 2)
        self._grid_edges = torch.unique(self.all_edges, dim=0)
        self.register_buffer("grid_edges", self._grid_edges)

        self.x_nx3 = self.x_nx3.detach()
        self.cube_fx8 = self.cube_fx8.detach()
        self.grid_edges = self.grid_edges.detach()

    def _init_weights(self):
        # 先使用线性层默认(或其它自定义)初始化
        # 例如 xavier, normal_ 等
        nn.init.xavier_normal_(self.fc_params_predictor.weight)
        nn.init.zeros_(self.fc_params_predictor.bias)

        # 然后将后 (3 + 21) = 24 个输出通道的权重和偏置全部置为 0
        with torch.no_grad():
            # 权重的形状是 [out_features, in_features]
            # 前 1 个输出通道是第 0 行，后 24 个输出通道是第 1~24 行
            self.fc_params_predictor.weight[1:] = 0.0

            # 偏置的形状是 [out_features]
            # 将后 24 个的偏置也设为 0
            self.fc_params_predictor.bias[1:] = 0.0

            # self.fc_params_predictor.weight.fill_(0.0)
            # b_sdf = torch.empty((1,)).uniform_(-0.1, 0.9)
            # self.fc_params_predictor.bias.copy_(b_sdf)


    def encode(self, pc, pc_feat):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs

        ###### fps
        flattened = pc.view(B * N, D)
        feat = pc_feat.view(B * N, -1)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        offset = torch.tensor([i * N for i in range(B + 1)], dtype=torch.int32, device=pc.device).contiguous()
        ### compute neighbor index
        reference_index, _ = pointops.knn_query(16, pos, offset)

        # num_inputs：输入点数
        # num_latents：期望降维点数
        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_feat = feat[idx]

        ######
        # FPS下采样得到M个点
        M = sampled_pc.shape[0] // B
        sampled_offset = torch.tensor([i * M for i in range(B + 1)], dtype=torch.int32, device=pc.device).contiguous()

        points = [pos, feat, offset]
        sampled_points = [sampled_pc, sampled_feat, sampled_offset]

        # 融合了坐标和特征的下采样点特征
        _, sampled_pc_embeddings, _ = self.point_embed(sampled_points)
        # 融合了坐标和特征的原始点特征
        _, pc_embeddings, _ = self.point_embed(points)

        sampled_pc_embeddings = sampled_pc_embeddings.view(B, M, -1)
        pc_embeddings = pc_embeddings.view(B, N, -1)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        for self_attn, self_ff in self.encoder_layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # x: [B, M, C]
        # latent_dim = C0 (32)
        # self.mean_fc = nn.Linear(dim, latent_dim)
        # self.logvar_fc = nn.Linear(dim, latent_dim)
        mean = self.mean_fc(x)  # [B, M, C] -> [B, M, C0]
        logvar = self.logvar_fc(x)  # [B, M, C] -> [B, M, C0]

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x

    # [B, 129^3, 21] -> [B, 128^3, 21]
    def get_cube_weight(self, pred_point_weight):
        # 使用高级索引，得到形状 [B, C, 8, 21]
        grid_points = pred_point_weight[:, self.cube_fx8, :]
        # 在维度 2 上求平均，结果形状为 [B, C, 21]
        grid_features = grid_points.mean(dim=2)
        return grid_features

    def get_fc_output(self, pred_sdf, pred_deform, cube_weight):
        batch = pred_sdf.shape[0]
        list_vertices = []
        list_faces = []
        list_losses = []
        for b in range(batch):
            sdf = pred_sdf[b, :]
            # print(f"sdf min = {sdf.min()}, sdf max = {sdf.max()}")
            deform = pred_deform[b, :, :]
            weight = cube_weight[b, :, :]
            # grid_verts = self.x_nx3 + (1.0 - 1e-8) / (self.voxel_grid_res * 2) * torch.tanh(deform)
            grid_verts = self.x_nx3 + (1.0 - 1e-8) / (self.voxel_grid_res * 2) * torch.tanh(deform)
            vertices, faces, L_dev = self.fc(grid_verts, sdf, self.cube_fx8, self.voxel_grid_res, beta_fx12=weight[:, :12],
                                        alpha_fx8=weight[:, 12:20],
                                        gamma_f=weight[:, 20], training=self.training)
            list_vertices.append(vertices)
            list_faces.append(faces)
            list_losses.append(L_dev)

        # batch_vertices = torch.stack(list_vertices, dim=0)
        # batch_faces = torch.stack(list_faces, dim=0)

        # [B*C]
        L_dev = torch.cat(list_losses).mean()

        return list_vertices, list_faces, L_dev

    # def get_random_cam(self, batch_size, fovy = np.deg2rad(45), iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=1.5, device="cuda"):
    #     def get_random_camera():
    #         proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])
    #         mv     = util.translate(0, 0, -cam_radius) @ util.random_rotation_translation(0.125)
    #         mvp    = proj_mtx @ mv
    #         return mv, mvp
    #     mv_batch = []
    #     mvp_batch = []
    #     for i in range(batch_size):
    #         mv, mvp = get_random_camera()
    #         mv_batch.append(mv)
    #         mvp_batch.append(mvp)
    #     return torch.stack(mv_batch).to(device), torch.stack(mvp_batch).to(device)
    #     # return mv_batch, mvp_batch

    def get_center_boundary_index(self, grid_res, device):
        v = torch.zeros((grid_res + 1, grid_res + 1, grid_res + 1), dtype=torch.bool, device=device)
        v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = True
        center_indices = torch.nonzero(v.reshape(-1))

        v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = False
        v[:2, ...] = True
        v[-2:, ...] = True
        v[:, :2, ...] = True
        v[:, -2:, ...] = True
        v[:, :, :2] = True
        v[:, :, -2:] = True
        boundary_indices = torch.nonzero(v.reshape(-1))
        return center_indices, boundary_indices

    def postprocess_sdf(self, sdf, grid_res, center_indices, boundary_indices):
        """
        对预测的 sdf 进行后处理，确保在出现空形状（全部正或全部负）时，
        强制在网格中产生正负交界。

        Args:
            sdf (torch.Tensor): 形状 [B, V] 的 SDF，V = (grid_res+1)^3。
            grid_res (int): 网格分辨率。
            center_indices (torch.Tensor): 网格中中心顶点的索引，形状 [num_center, 1]。
            boundary_indices (torch.Tensor): 网格中边界顶点的索引，形状 [num_boundary, 1]。

        Returns:
            torch.Tensor: 后处理后的 sdf，形状依然为 [B, V]。
        """
        B = sdf.shape[0]
        V_expected = (grid_res + 1) ** 3
        assert sdf.shape[1] == V_expected, f"Expected {V_expected} elements, got {sdf.shape[1]}"

        # 将 sdf reshape 成 [B, grid_res+1, grid_res+1, grid_res+1]
        sdf_grid = sdf.reshape(B, grid_res + 1, grid_res + 1, grid_res + 1)
        # 取内部体素（剔除边界），形状为 [B, N_inner]
        sdf_inner = sdf_grid[:, 1:-1, 1:-1, 1:-1].reshape(B, -1)

        # 统计内部正值和负值的数量
        pos_count = (sdf_inner > 0).sum(dim=-1)
        neg_count = (sdf_inner < 0).sum(dim=-1)

        # 判断是否出现空形状：即内部全部正或全部负
        zero_surface = (pos_count == 0) | (neg_count == 0)  # [B] 的布尔张量

        if torch.sum(zero_surface).item() > 0:
            # 构造一个 update_sdf，用于修正空形状
            update_sdf = torch.zeros_like(sdf[0:1])  # [1, V]
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            # 注意 center_indices 和 boundary_indices 的 shape 是 [N, 1]，需 squeeze 以获得一维索引
            center_idx = center_indices.squeeze(-1)
            boundary_idx = boundary_indices.squeeze(-1)

            # 对中心顶点加上 (1.0 - min_sdf) 使其变为正（或者至少大于0）
            update_sdf[:, center_idx] += (1.0 - min_sdf)
            # 对边界顶点加上 (-1 - max_sdf) 使其变为负
            update_sdf[:, boundary_idx] += (-1 - max_sdf)

            new_sdf = torch.zeros_like(sdf)
            for i in range(B):
                if zero_surface[i]:
                    new_sdf[i:i + 1] = update_sdf  # 对于该 batch 用更新值
            # 构造一个更新掩码，0的位置表示需要更新
            update_mask = (new_sdf == 0).float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)

            # 对于被更新的 batch，将其 sdf detach（使之不参与后续梯度更新）
            sdf_list = []
            for i in range(B):
                if zero_surface[i]:
                    sdf_list.append(sdf[i:i + 1].detach())
                else:
                    sdf_list.append(sdf[i:i + 1])
            sdf = torch.cat(sdf_list, dim=0)
        return sdf


    def get_predicted_fc(self, fc_latents):
        batch = fc_latents.shape[0]
        pred_sdf = fc_latents[:, :, 0] # [B, 129^3]

        center_indices, boundary_indices = self.get_center_boundary_index(self.voxel_grid_res, fc_latents.device)
        pred_sdf = self.postprocess_sdf(pred_sdf, self.voxel_grid_res, center_indices, boundary_indices)
        pred_deform = fc_latents[:, :, 1: 4] # [B, 129^3, 3]
        pred_weight = fc_latents[:, :, 4:] # [B, 129^3, 21]

        cube_weight = self.get_cube_weight(pred_weight) # [B, 128^3, 21]

        list_vertices, list_faces, L_dev = self.get_fc_output(pred_sdf, pred_deform, cube_weight)

        list_flexicubes_mesh = []
        for i_mesh in range(len(list_vertices)):
            flexicubes_mesh = Mesh(list_vertices[i_mesh], list_faces[i_mesh])
            list_flexicubes_mesh.append(flexicubes_mesh)
        del list_vertices, list_faces
        return list_flexicubes_mesh, L_dev, pred_sdf, cube_weight

    def decode(self, x):
        # [B, M, C0] -> [B, M, C]

        B, M, D = x.shape
        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        # queries_embeddings = self.point_embed(queries)

        # [B, 129^3, 3]
        queries = self.x_nx3.to(x.device, non_blocking=True)
        queries = queries.repeat(B, 1, 1)
        queries_embeddings = self.point_pos_embed(queries) # [B, M, C(512)]
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        fc_params = self.fc_params_predictor(latents)

        list_flexicubes_mesh, L_dev, pred_sdf, cube_weight = self.get_predicted_fc(fc_params)

        # return self.to_outputs(latents)
        return list_flexicubes_mesh, L_dev, pred_sdf, cube_weight

    def forward(self, pc, pc_feat):
        kl, x = self.encode(pc, pc_feat)

        list_flexicubes_mesh, L_dev, pred_sdf, cube_weight = self.decode(x)

        # return o.squeeze(-1), kl
        return {
                'list_mesh': list_flexicubes_mesh,
                'kl': kl, 'sdf': pred_sdf,
                'grid_edges': self.grid_edges,
                'weight': cube_weight,
                'L_dev': L_dev
                }



def create_autoencoder(dim=512, M=512, latent_dim=64, N=2048, determinisitc=False):
    # if determinisitc:
    #     model = AutoEncoder(
    #         depth=24,
    #         dim=dim,
    #         queries_dim=dim,
    #         output_dim = 1,
    #         num_inputs = N,
    #         num_latents = M,
    #         heads = 8,
    #         dim_head = 64,
    #     )
    # else:
    model = KLAutoEncoder(
        depth=24,
        dim=dim,
        queries_dim=dim,
        output_dim = 1,
        num_inputs = N,
        num_latents = M,
        latent_dim = latent_dim, # C0 = 32
        heads = 8,
        dim_head = 64,
    )
    return model

def kl_d512_m512_l32(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, determinisitc=False)