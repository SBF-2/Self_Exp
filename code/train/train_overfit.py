"""
过拟合训练脚本 - 验证模型和代码正确性
=============================================

目标：
1. 收集128条训练数据（使用原始3个Atari环境）
2. 用这128条数据循环训练模型直到损失<20
3. 训练过程中保存每个环境下损失最小的4张对照图
4. 保存过拟合模型，验证训练流程正确性
5. 所有文件名加overfit前缀，不覆盖原有文件
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from collections import defaultdict

# Import your custom modules
from model.PPM_attention import PredictiveRepModel
from env.AtariEnv_random import AtariEnvManager

# --- Configuration ---
# Environment parameters (使用原始3个环境)
NUM_GAMES_TO_SELECT_FROM = 3  # 原始3个游戏环境
NUM_ENVS = 128                # 128个并行环境收集数据
FIXED_DATA_SIZE = 128         # 固定使用128条数据
IMG_H, IMG_W, IMG_C = 210, 160, 3

# Model parameters
LATENT_DIM = 256
ENCODER_LAYERS = [2, 2]
DECODER_LAYERS = [2, 2, 1]

# Training parameters
LEARNING_RATE = 1e-3          # 稍高的学习率加速过拟合
TARGET_LOSS = 20.0            # 目标损失阈值
MAX_EPOCHS = 40000             # 最大训练轮数
PRINT_INTERVAL = 200         # 打印间隔
SAVE_INTERVAL = 5000           # 保存间隔
EVAL_INTERVAL = 500          # 评估并保存对比图间隔
SEQ_LEN = 1

# Overfit experiment paths (所有文件名加overfit前缀)
OVERFIT_MODEL_DIR = "train_models_attention/overfit_trained_models"
OVERFIT_DATA_FILE = "overfit_training_data_128.pkl"
OVERFIT_LOG_FILE = "log/overfit_training.log"
OVERFIT_IMAGES_DIR = "overfit_training_images"  # 训练过程中的对比图
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_observations(obs_list, device):
    """
    Converts a list of observations (H, W, C) uint8 to a tensor (B, C, H, W) float.
    """
    processed_obs = []
    for obs_hwc in obs_list:
        obs_chw = np.transpose(obs_hwc, (2, 0, 1)) # HWC to CHW
        processed_obs.append(obs_chw)
    
    # Stack, normalize to [0, 1], convert to tensor
    obs_tensor_bchw = torch.tensor(np.array(processed_obs), dtype=torch.float32, device=device)
    return obs_tensor_bchw

def denormalize_image(img_tensor):
    """Convert tensor back to displayable format"""
    # Clamp values to [0, 255] range and convert to uint8
    img_np = torch.clamp(img_tensor, 0, 255).cpu().numpy().astype(np.uint8)
    return np.transpose(img_np, (1, 2, 0))  # CHW to HWC

def create_comparison_image(real_img, pred_img, sample_idx, loss_value, env_name, epoch):
    """Create a side-by-side comparison image during training"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Real image (left)
    axes[0].imshow(real_img)
    axes[0].set_title('Real Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Predicted image (right)
    axes[1].imshow(pred_img)
    axes[1].set_title('Generated Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overall title
    fig.suptitle(f'{env_name} - Training Epoch {epoch}\nSample {sample_idx} - MSE Loss: {loss_value:.6f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def classify_samples_by_environment(dataset):
    """
    根据图像特征或其他信息将样本分类到不同环境
    由于我们没有显式的环境标签，这里使用简单的哈希方法来模拟环境分类
    """
    env_samples = defaultdict(list)
    
    for idx, data in enumerate(dataset):
        # 使用obs1的哈希值来模拟环境分类
        # 在真实情况下，你可以根据游戏画面特征或其他信息来分类
        obs_hash = hash(data['obs1'].tobytes()) % 3  # 分成3个环境
        env_name = ['Seaquest-v4', 'Riverraid-v4', 'ChopperCommand-v4'][obs_hash]
        env_samples[env_name].append(idx)
    
    return env_samples

def evaluate_and_save_best_samples(model, dataset, env_samples, epoch, images_dir):
    """
    评估模型并保存每个环境下损失最小的4张对照图
    """
    model.eval()
    
    # 准备批次数据
    obs1_input, actions_input, obs2_target = create_batch_from_dataset(dataset, DEVICE)
    
    with torch.no_grad():
        # 模型推理
        _, predicted_obs2_raw = model(obs1_input, actions_input)
        
        # 处理输出
        predicted_obs2_processed = predicted_obs2_raw.squeeze(1)  # (B, C, H, W)
        obs2_target_processed = obs2_target.squeeze(1)  # (B, C, H, W)
        
        # 裁剪预测结果
        predicted_obs2_cropped = model._crop_output(predicted_obs2_processed)
        
        # 计算每个样本的损失
        individual_losses = []
        for i in range(predicted_obs2_cropped.shape[0]):
            loss_i = F.mse_loss(
                predicted_obs2_cropped[i:i+1], 
                obs2_target_processed[i:i+1]
            ).item()
            individual_losses.append(loss_i)
    
    # 为每个环境保存最佳4张对比图
    total_saved = 0
    for env_name, sample_indices in env_samples.items():
        if len(sample_indices) == 0:
            continue
        
        # 获取该环境下所有样本的损失
        env_losses = [(idx, individual_losses[idx]) for idx in sample_indices]
        env_losses.sort(key=lambda x: x[1])  # 按损失排序
        
        # 取前4个最佳样本
        best_samples = env_losses[:min(4, len(env_losses))]
        
        # 创建环境特定的目录
        env_dir = os.path.join(images_dir, f"epoch_{epoch}", env_name.replace('-v4', ''))
        os.makedirs(env_dir, exist_ok=True)
        
        # 保存最佳样本的对比图
        for rank, (sample_idx, loss_val) in enumerate(best_samples):
            real_img = dataset[sample_idx]['obs2']
            pred_img = denormalize_image(predicted_obs2_cropped[sample_idx])
            
            fig = create_comparison_image(
                real_img, pred_img, sample_idx + 1, loss_val, env_name, epoch
            )
            
            save_path = os.path.join(env_dir, f'best_{rank+1}_sample_{sample_idx+1}_loss_{loss_val:.6f}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            total_saved += 1
        
        print(f"    {env_name}: 保存了 {len(best_samples)} 张最佳对比图")
    
    model.train()  # 恢复训练模式
    return np.mean(individual_losses), total_saved

def collect_fixed_dataset(env_manager, target_size=128):
    """收集固定大小的数据集"""
    print(f"开始收集 {target_size} 条固定训练数据...")
    
    collected_data = []
    step_count = 0
    
    while len(collected_data) < target_size:
        # Sample actions and step environments
        actions_list = env_manager.sample_actions()
        experiences = env_manager.step(actions_list)
        
        # Process experiences
        for exp in experiences:
            if len(collected_data) >= target_size:
                break
                
            obs1, action, obs2, reward, done = exp
            collected_data.append({
                'obs1': obs1.copy(),
                'obs2': obs2.copy(), 
                'action': action,
                'reward': reward,
                'done': done
            })
        
        step_count += 1
        if step_count % 10 == 0:
            print(f"  收集进度: {len(collected_data)}/{target_size} 条数据")
    
    print(f"数据收集完成！共收集 {len(collected_data)} 条数据")
    return collected_data[:target_size]  # 确保精确数量

def save_dataset(dataset, filepath):
    """保存数据集到文件"""
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"数据集已保存到: {filepath}")

def create_batch_from_dataset(dataset, device):
    """从数据集创建批次数据"""
    obs1_list = [data['obs1'] for data in dataset]
    obs2_list = [data['obs2'] for data in dataset] 
    actions_list = [data['action'] for data in dataset]
    
    # Preprocess observations
    obs1_batch_bchw = preprocess_observations(obs1_list, device)
    obs2_target_batch_bchw = preprocess_observations(obs2_list, device)
    
    # Reshape for model input (B, L, C, H, W)
    obs1_input_blchw = obs1_batch_bchw.unsqueeze(1)
    obs2_target_blchw = obs2_target_batch_bchw.unsqueeze(1)
    
    # Actions to tensor (B, L)
    actions_tensor_bl = torch.tensor(actions_list, dtype=torch.long, device=device).unsqueeze(1)
    
    return obs1_input_blchw, actions_tensor_bl, obs2_target_blchw

def main():
    print(f"开始过拟合实验 - 设备: {DEVICE}")
    print("="*60)
    
    # Create directories
    os.makedirs(OVERFIT_MODEL_DIR, exist_ok=True)
    os.makedirs(OVERFIT_IMAGES_DIR, exist_ok=True)
    
    # Initialize log file
    with open(OVERFIT_LOG_FILE, 'w') as f:
        f.write("过拟合训练日志\n")
        f.write("="*50 + "\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"目标: 在128条数据上过拟合，损失降至{TARGET_LOSS}以下\n")
        f.write(f"设备: {DEVICE}\n")
        f.write(f"训练过程中将保存每个环境下损失最小的4张对照图\n")
        f.write("="*50 + "\n\n")
    
    # --- Step 1: 收集固定数据集 ---
    print("步骤 1: 收集固定训练数据集")
    
    if os.path.exists(OVERFIT_DATA_FILE):
        print(f"发现已存在的数据文件: {OVERFIT_DATA_FILE}")
        response = input("是否重新收集数据？(y/n): ").lower().strip()
        if response != 'y':
            print("加载已存在的数据集...")
            with open(OVERFIT_DATA_FILE, 'rb') as f:
                fixed_dataset = pickle.load(f)
            print(f"已加载 {len(fixed_dataset)} 条数据")
        else:
            # 重新收集数据
            print("重新收集数据...")
            env_manager = AtariEnvManager(
                num_games=NUM_GAMES_TO_SELECT_FROM, 
                num_envs=NUM_ENVS, 
                render_mode=None
            )
            fixed_dataset = collect_fixed_dataset(env_manager, FIXED_DATA_SIZE)
            save_dataset(fixed_dataset, OVERFIT_DATA_FILE)
            env_manager.close()
    else:
        # 收集新数据
        env_manager = AtariEnvManager(
            num_games=NUM_GAMES_TO_SELECT_FROM, 
            num_envs=NUM_ENVS, 
            render_mode=None
        )
        fixed_dataset = collect_fixed_dataset(env_manager, FIXED_DATA_SIZE)
        save_dataset(fixed_dataset, OVERFIT_DATA_FILE)
        env_manager.close()
    
    # --- Step 2: 初始化模型 ---
    print("\n步骤 2: 初始化模型")
    
    # 从第一个数据样本获取动作维度
    sample_env_manager = AtariEnvManager(num_games=NUM_GAMES_TO_SELECT_FROM, num_envs=1, render_mode=None)
    action_dim = sample_env_manager.envs[0].action_space.n
    sample_env_manager.close()
    
    model = PredictiveRepModel(
        img_in_channels=IMG_C,
        img_out_channels=IMG_C,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers_config=DECODER_LAYERS,
        target_img_h=IMG_H,
        target_img_w=IMG_W,
        action_dim=action_dim,
        latent_dim=LATENT_DIM
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"模型初始化完成，动作维度: {action_dim}")
    
    # --- Step 3: 过拟合训练 ---
    print(f"\n步骤 3: 开始过拟合训练 (目标损失 < {TARGET_LOSS})")
    print("="*60)
    
    # 准备批次数据
    obs1_input, actions_input, obs2_target = create_batch_from_dataset(fixed_dataset, DEVICE)
    
    # 按环境分类样本，用于保存最佳对比图
    print("按环境分类样本...")
    env_samples = classify_samples_by_environment(fixed_dataset)
    for env_name, indices in env_samples.items():
        print(f"  {env_name}: {len(indices)} 个样本")
        with open(OVERFIT_LOG_FILE, 'a') as f:
            f.write(f"环境 {env_name}: {len(indices)} 个样本\n")
    
    epoch = 0
    best_loss = float('inf')
    
    while epoch < MAX_EPOCHS:
        epoch += 1
        
        # 训练一个epoch
        loss = model.train_on_batch(obs1_input, actions_input, obs2_target, optimizer)
        
        if loss < best_loss:
            best_loss = loss
        
        # 打印进度
        if epoch % PRINT_INTERVAL == 0:
            log_message = f"Epoch {epoch:4d}: Loss = {loss:.6f}, Best = {best_loss:.6f}"
            print(log_message)
            
            # 记录到日志文件
            with open(OVERFIT_LOG_FILE, 'a') as f:
                f.write(f"{log_message}\n")
        
        # 定期评估并保存最佳样本对比图
        if epoch % EVAL_INTERVAL == 0:
            print(f"  评估模型并保存最佳样本对比图 (Epoch {epoch})...")
            avg_eval_loss, saved_images = evaluate_and_save_best_samples(
                model, fixed_dataset, env_samples, epoch, OVERFIT_IMAGES_DIR
            )
            print(f"  评估平均损失: {avg_eval_loss:.6f}, 保存了 {saved_images} 张对比图")
            
            with open(OVERFIT_LOG_FILE, 'a') as f:
                f.write(f"Epoch {epoch} 评估: 平均损失={avg_eval_loss:.6f}, 保存图片={saved_images}张\n")
        
        # 保存检查点
        if epoch % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(OVERFIT_MODEL_DIR, f'overfit_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            
            with open(OVERFIT_LOG_FILE, 'a') as f:
                f.write(f"保存检查点: {checkpoint_path}\n")
        
        # 检查是否达到目标损失
        if loss < TARGET_LOSS:
            print(f"\n🎉 达到目标损失！")
            print(f"最终损失: {loss:.6f} < {TARGET_LOSS}")
            print(f"训练轮数: {epoch}")
            
            # 最终评估并保存最佳样本
            print("进行最终评估并保存最佳样本...")
            final_avg_loss, final_saved_images = evaluate_and_save_best_samples(
                model, fixed_dataset, env_samples, epoch, OVERFIT_IMAGES_DIR
            )
            print(f"最终评估平均损失: {final_avg_loss:.6f}")
            
            # 保存最终模型
            final_model_path = os.path.join(OVERFIT_MODEL_DIR, 'overfit128.pth')
            torch.save(model.state_dict(), final_model_path)
            print(f"最终模型已保存: {final_model_path}")
            
            # 记录成功信息
            with open(OVERFIT_LOG_FILE, 'a') as f:
                f.write(f"\n训练成功完成！\n")
                f.write(f"最终训练损失: {loss:.6f}\n")
                f.write(f"最终评估损失: {final_avg_loss:.6f}\n")
                f.write(f"训练轮数: {epoch}\n")
                f.write(f"最终模型: {final_model_path}\n")
                f.write(f"保存的对比图总数: {final_saved_images}\n")
                f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            break
    
    if epoch >= MAX_EPOCHS:
        print(f"\n⚠️  达到最大训练轮数 ({MAX_EPOCHS})，未达到目标损失")
        print(f"当前最佳损失: {best_loss:.6f}")
        
        # 进行最终评估
        print("进行最终评估并保存最佳样本...")
        final_avg_loss, final_saved_images = evaluate_and_save_best_samples(
            model, fixed_dataset, env_samples, epoch, OVERFIT_IMAGES_DIR
        )
        print(f"最终评估平均损失: {final_avg_loss:.6f}")
        
        # 仍然保存模型
        final_model_path = os.path.join(OVERFIT_MODEL_DIR, 'overfit128_max_epochs.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"模型已保存: {final_model_path}")
        
        with open(OVERFIT_LOG_FILE, 'a') as f:
            f.write(f"\n训练达到最大轮数限制\n")
            f.write(f"最佳训练损失: {best_loss:.6f}\n")
            f.write(f"最终评估损失: {final_avg_loss:.6f}\n")
            f.write(f"模型保存: {final_model_path}\n")
            f.write(f"保存的对比图总数: {final_saved_images}\n")
    
    print("\n" + "="*60)
    print("过拟合训练完成！")
    print(f"训练数据: {OVERFIT_DATA_FILE}")
    print(f"模型保存: {OVERFIT_MODEL_DIR}")
    print(f"训练日志: {OVERFIT_LOG_FILE}")
    print(f"对比图像: {OVERFIT_IMAGES_DIR}")
    print("="*60)

if __name__ == '__main__':
    main()