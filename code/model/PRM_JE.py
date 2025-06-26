# PRM_JE.py - Simplified PRM with only Encoder and Predictive Model
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import os
import math


# 假设 .residual 包含 DownBlock2d 和 UpBlock2d。
# 为了独立测试，如果不可用，我们定义增强版本。
try:
    from .residual import DownBlock2d, UpBlock2d
except ImportError:

    # --- SE注意力模块 ---
    class SEBlock(nn.Module):
        """Squeeze-and-Excitation Block"""
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c = x.size(0), x.size(1)
            y = self.squeeze(x).view(b, c)
            y = self.excitation(y).view(b, c, 1, 1)
            return x * y

    # --- 增强残差块 ---
    class EnhancedResidualBlock(nn.Module):
        """增强的残差块，包含SE注意力"""
        def __init__(self, channels, num_layers=2, use_se=True):
            super().__init__()
            self.use_se = use_se

            layers = []
            for i in range(num_layers):
                layers.extend([
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                ])
                if i < num_layers - 1:
                    layers.append(nn.ReLU(inplace=True))

            self.conv_layers = nn.Sequential(*layers)

            if self.use_se:
                self.se = SEBlock(channels)

            self.final_relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = x
            out = self.conv_layers(x)

            if self.use_se:
                out = self.se(out)

            out = out + identity
            out = self.final_relu(out)
            return out

    class DownBlock2d(nn.Module):
        """下采样块"""
        def __init__(self, in_channels, out_channels, num_layers, downsample=False, use_se=True):
            super().__init__()
            self.downsample = downsample
            stride = 2 if downsample else 1

            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            self.residual_blocks = nn.ModuleList([
                EnhancedResidualBlock(out_channels, 2, use_se) for _ in range(num_layers)
            ])

            self.shortcut = nn.Identity()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            identity = self.shortcut(x)
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)

            if self.downsample and identity.shape[2:] != out.shape[2:]:
                identity = F.adaptive_avg_pool2d(identity, out.shape[2:])

            out = out + identity

            for res_block in self.residual_blocks:
                out = res_block(out)

            return out


# --- 动作编码器 ---
class ActionEncoder(nn.Module):
    """动作编码器，将0-18的整数动作编码为向量"""
    def __init__(self, action_dim=19, embed_dim=128, output_dim=256):  # 0-18共19个动作
        super().__init__()
        self.action_embedding = nn.Embedding(action_dim, embed_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, actions):
        """
        Args:
            actions: (B,) 动作索引
        Returns:
            action_features: (B, output_dim) 动作特征
        """
        embedded = self.action_embedding(actions.long())  # (B, embed_dim)
        return self.action_proj(embedded)  # (B, output_dim)


# --- Transformer层 ---
class TransformerLayer(nn.Module):
    """单个Transformer层，使用动作作为query的交叉注意力机制"""
    
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        
    def forward(self, x, action_features):
        """
        Args:
            x: 视觉特征 (B, d_model)
            action_features: 动作特征 (B, d_model)
        Returns:
            输出特征 (B, d_model)
        """
        B = x.size(0)
        
        # 1. Multi-head attention with residual connection
        residual = x
        x_norm = self.ln1(x)
        action_norm = self.ln1(action_features)
        
        # 动作作为query, 视觉特征作为key和value
        Q = self.q_proj(action_norm).view(B, self.n_heads, self.head_dim)  # (B, H, D)
        K = self.k_proj(x_norm).view(B, self.n_heads, self.head_dim)       # (B, H, D)
        V = self.v_proj(x_norm).view(B, self.n_heads, self.head_dim)       # (B, H, D)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, 1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力
        attended = torch.matmul(attn_weights, V)  # (B, H, D)
        attended = attended.view(B, self.d_model)  # (B, d_model)
        
        # 输出投影和残差连接
        attended = self.out_proj(attended)
        x = residual + attended
        
        # 2. Feed forward with residual connection
        residual = x
        x_norm = self.ln2(x)
        x = residual + self.ffn_dropout(self.ffn(x_norm))
        
        return x


# --- 三层Transformer预测模型 ---
class PredictiveModel(nn.Module):
    """三层Transformer预测模型"""
    
    def __init__(self, d_model=256, n_heads=8, n_layers=3, d_ff=1024, dropout=0.1, action_dim=19):
        super().__init__()
        
        # 动作编码器
        self.action_encoder = ActionEncoder(action_dim=action_dim, output_dim=d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # 输出层归一化
        self.output_ln = nn.LayerNorm(d_model)
        
    def forward(self, x, actions):
        """
        Args:
            x: 编码器输出 (B, d_model)
            actions: 动作索引 (B,)
        Returns:
            预测特征 (B, d_model)
        """
        # 编码动作
        action_features = self.action_encoder(actions)  # (B, d_model)
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x, action_features)
        
        # 输出归一化
        x = self.output_ln(x)
        
        return x


# --- 编码器（重命名为encoder）---
class Encoder(nn.Module):
    """编码器，输入210*160*3图像"""

    def __init__(self, in_channels: int = 3, layers: List[int] = [2, 3, 4, 3], base_channels: int = 64) -> None:
        super().__init__()

        # Stem层 - 处理210*160*3输入
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=2, padding=1, bias=False),  # 105*80
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 特征提取网络
        self.layer1 = DownBlock2d(base_channels, base_channels, layers[0], downsample=False)       # 105*80
        self.layer2 = DownBlock2d(base_channels, base_channels * 2, layers[1], downsample=True)    # 53*40
        self.layer3 = DownBlock2d(base_channels * 2, base_channels * 4, layers[2], downsample=True) # 27*20
        self.layer4 = DownBlock2d(base_channels * 4, base_channels * 8, layers[3], downsample=True) # 14*10

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(base_channels * 8, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, 3, 210, 160)
        Returns:
            features: 特征向量 (B, 256)
        """
        # 确保输入尺寸正确
        assert x.shape[-2:] == (210, 160), f"Expected input size (210, 160), got {x.shape[-2:]}"

        x = self.stem(x)  # (B, 64, 105, 80)
        x = self.layer1(x)  # (B, 64, 105, 80)
        x = self.layer2(x)  # (B, 128, 53, 40)
        x = self.layer3(x)  # (B, 256, 27, 20)
        x = self.layer4(x)  # (B, 512, 14, 10)

        # 全局池化和特征投影
        pooled = self.global_pool(x)  # (B, 512, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # (B, 512)
        features = self.feature_proj(flattened)  # (B, 256)

        return features


# --- PRM_JE主模型 ---
class PRM_JE(nn.Module):
    """Predictive Representation Model - Just Encoder (只包含编码器和预测模型)"""
    
    def __init__(
        self, 
        img_in_channels: int = 3,
        encoder_layers: List[int] = [2, 3, 4, 3],
        action_dim: int = 19,  # 0-18共19个动作
        latent_dim: int = 256,
        base_channels: int = 64, 
        num_attention_heads: int = 8,
        transformer_layers: int = 3,
        loss_weight: float = 1.0,  # 只有一个损失权重
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # 编码器
        self.encoder = Encoder(img_in_channels, encoder_layers, base_channels)
        
        # 三层Transformer预测模型
        self.predictive_model = PredictiveModel(
            d_model=latent_dim,
            n_heads=num_attention_heads,
            n_layers=transformer_layers,
            action_dim=action_dim,
            dropout=dropout
        )
        
        self.loss_weight = loss_weight
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """参考LLM的权重初始化策略"""
        if isinstance(m, nn.Linear):
            # Xavier uniform for linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            # 正态分布初始化embedding
            nn.init.normal_(m.weight, meas zn=0, std=0.02)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Kaiming初始化卷积层
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, images: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            images: 输入图像 (B, 3, 210, 160)
            actions: 动作 (B,) - 0到18的整数
        Returns:
            encoder_features: 编码器输出 (B, 256)
            predicted_features: 预测模型输出 (B, 256)
        """
        # 确保动作在正确范围内
        assert actions.min() >= 0 and actions.max() <= 18, f"Actions must be in range [0, 18], got [{actions.min()}, {actions.max()}]"
        
        # 编码
        encoder_features = self.encoder(images)  # (B, 256)
        # Transformer预测
        predicted_features = self.predictive_model(encoder_features, actions)  # (B, 256)
        
        return encoder_features, predicted_features
    
    def compute_loss(self, 
                     predicted_features: torch.Tensor,
                     target_features: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        计算特征预测损失
        Args:
            predicted_features: 预测特征 (B, 256)
            target_features: 目标特征 (B, 256)
        """
        # Loss_vector: 余弦相似度损失 - 批次内平均
        pred_norm = F.normalize(predicted_features, p=2, dim=1)
        target_norm = F.normalize(target_features, p=2, dim=1)
        cosine_sim = (pred_norm * target_norm).sum(dim=1)
        loss_vector = (1.0 - cosine_sim).mean()  # 余弦距离的平均
        
        # 总损失
        total_loss = self.loss_weight * loss_vector
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'loss_vector': loss_vector.item(),
            'cosine_similarity': cosine_sim.mean().item()
        }
        
        return total_loss, loss_dict

    def train_on_batch(self, 
                      batch_images: torch.Tensor,
                      batch_actions: torch.Tensor,
                      batch_next_images: torch.Tensor,
                      optimizer: torch.optim.Optimizer) -> dict:
        """
        在单个batch上训练
        Args:
            batch_images: 当前图像 [I1,I2,I3...] (B, 3, 210, 160)
            batch_actions: 动作 (B,)
            batch_next_images: 下一帧图像 (B, 3, 210, 160) 
            optimizer: 优化器
        """
        self.train()
        optimizer.zero_grad()
        
        batch_size = batch_images.size(0)
        if batch_size <= 1:
            raise ValueError("Batch size must be greater than 1")
        
        # 按照要求：对batch中前m个样本进行处理 (1<=m<=batch_size-1)
        # 使用I1,I2,...,Im预测I2',I3',...,I(m+1)'的特征
        input_images = batch_images[:-1]    # [I1,I2,...,Im]
        input_actions = batch_actions[:-1]  # 对应的动作
        target_next_images = batch_next_images[:-1]  # [I2,I3,...,I(m+1)] - 真实的下一帧
        
        # 前向传播：用[I1,I2,...,Im]和对应动作预测下一帧特征
        encoder_features, predicted_features = self.forward(input_images, input_actions)
        
        # 获取目标encoder特征 - 对真实的下一帧图像编码
        with torch.no_grad():
            target_encoder_features = self.encoder(target_next_images)  # [x2,x3,...,x(m+1)]
        
        # 计算损失
        total_loss, loss_dict = self.compute_loss(predicted_features, target_encoder_features)
        
        # 反向传播和优化
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss_dict

    def evaluate_on_batch(self,
                         batch_images: torch.Tensor,
                         batch_actions: torch.Tensor, 
                         batch_next_images: torch.Tensor) -> dict:
        """
        在单个batch上验证
        Args:
            batch_images: 当前图像 [I1,I2,I3...] (B, 3, 210, 160)
            batch_actions: 动作 (B,)
            batch_next_images: 下一帧图像 (B, 3, 210, 160)
        Returns:
            loss_dict: 包含各项损失的字典
        """
        self.eval()
        
        batch_size = batch_images.size(0)
        if batch_size <= 1:
            raise ValueError("Batch size must be greater than 1")
        
        with torch.no_grad():
            # 按照要求：对batch中前m个样本进行处理 (1<=m<=batch_size-1)
            input_images = batch_images[:-1]    # [I1,I2,...,Im]
            input_actions = batch_actions[:-1]  # 对应的动作
            target_next_images = batch_next_images[:-1]  # [I2,I3,...,I(m+1)]
            
            # 前向传播
            encoder_features, predicted_features = self.forward(input_images, input_actions)
            
            # 获取目标encoder特征
            target_encoder_features = self.encoder(target_next_images)  # [x2,x3,...,x(m+1)]
            
            # 计算损失
            total_loss, loss_dict = self.compute_loss(predicted_features, target_encoder_features)
            
        return loss_dict


def create_optimized_optimizer(model, base_lr=1e-4, weight_decay=1e-2):
    """
    创建优化的优化器配置，参考LLM训练最佳实践
    """
    # 分离不同类型的参数
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # LayerNorm, BatchNorm, bias等不进行权重衰减
        if len(param.shape) <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr
        }
    ]
    
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=base_lr,
        betas=(0.9, 0.95),  # 参考LLM训练的beta值
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    return optimizer


def create_lr_scheduler(optimizer, warmup_steps=1000, max_steps=10000, min_lr_ratio=0.1):
    """
    创建学习率调度器，包含warmup和cosine annealing
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup阶段
            return step / warmup_steps
        else:
            # Cosine annealing阶段
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


if __name__ == '__main__':
    # --- 模型测试 ---
    BATCH_SIZE = 8
    IMG_C = 3
    IMG_H = 210
    IMG_W = 160
    ACTION_DIM = 19  # 0-18共19个动作

    print("=== PRM_JE Model Testing ===")
    print(f"Input image size: {IMG_H}x{IMG_W}x{IMG_C}")
    print(f"Action range: 0-{ACTION_DIM-1}")
    print(f"Batch size: {BATCH_SIZE}")

    # 创建测试数据
    dummy_images = torch.rand(BATCH_SIZE, IMG_C, IMG_H, IMG_W)  # [0,1]范围
    dummy_actions = torch.randint(0, ACTION_DIM, (BATCH_SIZE,))  # 0-18的动作
    dummy_next_images = torch.rand(BATCH_SIZE, IMG_C, IMG_H, IMG_W)  # [0,1]范围

    # 创建模型
    model = PRM_JE(
        img_in_channels=IMG_C,
        encoder_layers=[2, 3, 4, 3],
        action_dim=ACTION_DIM,
        latent_dim=256,
        base_channels=64,
        num_attention_heads=8,
        transformer_layers=3,
        loss_weight=1.0,
        dropout=0.1
    )

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 测试前向传播
    print(f"\n=== 前向传播测试 ===")
    model.eval()
    with torch.no_grad():
        encoder_features, predicted_features = model(dummy_images, dummy_actions)
        print(f"✅ 编码器特征: {encoder_features.shape}")
        print(f"✅ 预测特征: {predicted_features.shape}")

    # 创建优化器
    optimizer = create_optimized_optimizer(model, base_lr=1e-4, weight_decay=1e-2)
    scheduler = create_lr_scheduler(optimizer, warmup_steps=100, max_steps=1000)

    # 测试训练
    print(f"\n=== 训练测试 ===")
    n_epochs = 50
    
    for epoch in range(n_epochs):
        # 训练
        loss_dict = model.train_on_batch(
            dummy_images, dummy_actions, dummy_next_images, optimizer
        )
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1:3d}/{n_epochs}, '
                  f'Total: {loss_dict["total_loss"]:.6f}, '
                  f'Cosine Sim: {loss_dict["cosine_similarity"]:.4f}, '
                  f'LR: {current_lr:.2e}')
            
            # 验证
            val_loss_dict = model.evaluate_on_batch(
                dummy_images, dummy_actions, dummy_next_images
            )
            print(f'    Validation: {val_loss_dict["total_loss"]:.6f}')

    print(f"\n=== PRM_JE模型总结 ===")
    print(f"✅ 编码器: 处理210×160×3图像 → 256维特征")
    print(f"✅ 动作编码: 0-18整数动作embedding")
    print(f"✅ 注意力: 动作作为query，视觉特征作为key/value")
    print(f"✅ 预测模型: 输出下一帧的特征表示")
    print(f"✅ 损失函数: 余弦相似度损失")
    print(f"✅ 优化: AdamW + warmup + 余弦退火")
    print(f"✅ 总参数: {total_params:,}")
    print("🎉 PRM_JE模型测试完成！")
