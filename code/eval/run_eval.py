"""
模型验证脚本
=============

该脚本用于验证训练好的PredictiveRepModel模型性能。

主要功能：
1. 扩展原始AtariEnvManager，添加第4个游戏环境（Breakout-v4）
2. 在4个不同的Atari游戏环境中收集验证数据
3. 加载训练好的模型权重并进行推理
4. 生成真实图像vs生成图像的对比可视化
5. 保存每个环境下最佳和最差的8个结果

注意：由于原始AtariEnv_random.py只包含3个游戏环境，本脚本通过
ExtendedAtariEnvManager类扩展了环境列表，添加了Breakout-v4作为第4个验证环境。
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import shutil
from datetime import datetime

# Import your custom modules
from model.PPM_attention import PredictiveRepModel
from env.AtariEnv_random import AtariEnvManager
import gymnasium as gym
import ale_py

# --- Configuration ---
# Model parameters
IMG_H, IMG_W, IMG_C = 210, 160, 3
LATENT_DIM = 256
ENCODER_LAYERS = [2, 2]
DECODER_LAYERS = [2, 2, 1]

# Validation parameters
NUM_VALIDATION_ENVS = 4  # Use 4 different environments
NUM_VALIDATION_STEPS = 100  # Collect 100 samples for validation
MODEL_CHECKPOINT_PATH = "/Users/feisong/Desktop/modeltrain/modeltraining/code/trained_models_attention/trained_models_attention/ppm_attention_step_20000.pth"
SAVE_DIR = "eval_res/imgs"
LOG_DIR = "log"
LOG_FILE = "log/eval_log.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extended Environment Manager for Validation
class ExtendedAtariEnvManager(AtariEnvManager):
    """扩展的Atari环境管理器，添加第4个游戏用于验证"""
    def __init__(self, num_games, render_mode=None, num_envs=1):
        # 扩展可用游戏列表，添加第4个游戏
        self.available_games = [
            "Seaquest-v4",
            "Riverraid-v4",
            "ChopperCommand-v4",
            "Breakout-v4"  # 新添加的第4个游戏
        ]
        
        # 确保请求的游戏数量不超过可用的游戏数量
        num_games = min(num_games, len(self.available_games))
        
        # 随机选择指定数量的游戏作为游戏池
        import numpy as np
        self.selected_games = np.random.choice(self.available_games, num_games, replace=False)
        print(f"验证游戏池中的游戏: {self.selected_games}")
        
        # 存储渲染模式配置
        self.render_mode = render_mode
        # 如果是 'human' 模式，只渲染第一个环境以避免窗口过多
        self.effective_render_mode = [render_mode] + [None] * (num_envs - 1) if render_mode == 'human' else [None] * num_envs
        
        # 为每个环境随机分配一个选中的游戏
        env_games = np.random.choice(self.selected_games, num_envs)
        print(f"尝试创建 {num_envs} 个验证环境实例")
        for i, game in enumerate(env_games):
            print(f"验证环境 {i+1} 初始游戏: {game}")

        self.envs = [] # 环境实例列表
        self.current_games = [] # 记录每个环境当前的游戏类型

        for i in range(num_envs):
            try:
                env = gym.make(env_games[i], render_mode=self.effective_render_mode[i])
                self.envs.append(env)
                self.current_games.append(env_games[i])
                print(f"验证环境实例 {i+1} '{env_games[i]}' 创建成功。动作空间: {env.action_space}")
            except Exception as e:
                print(f"创建验证环境实例 {i+1} '{env_games[i]}' 失败: {e}")
        
        if not self.envs:
            raise RuntimeError("未能成功创建任何验证环境实例。")

        self.num_envs = len(self.envs)
        self.current_obs = [None] * self.num_envs # 存储每个环境的当前观测
        self.episode_rewards = [0] * self.num_envs # 存储每个环境当前回合的奖励
        
        # 初始化所有环境的观测值
        for i in range(self.num_envs):
            obs, _ = self.envs[i].reset()
            self.current_obs[i] = obs

# Specific games for validation (ensuring we have exactly 4 different games)
VALIDATION_GAMES = [
    "Seaquest-v4",
    "Riverraid-v4", 
    "ChopperCommand-v4",
    "Breakout-v4"  # 新添加的第4个游戏
]

def preprocess_observations(obs_list, device):
    """
    Converts a list of observations (H, W, C) uint8 to a tensor (B, C, H, W) float.
    """
    processed_obs = []
    for obs_hwc in obs_list:
        obs_chw = np.transpose(obs_hwc, (2, 0, 1))  # HWC to CHW
        processed_obs.append(obs_chw)
    
    # Stack, normalize to [0, 1], convert to tensor
    obs_tensor_bchw = torch.tensor(np.array(processed_obs), dtype=torch.float32, device=device)
    return obs_tensor_bchw

def denormalize_image(img_tensor):
    """Convert tensor back to displayable format"""
    # Clamp values to [0, 255] range and convert to uint8
    img_np = torch.clamp(img_tensor, 0, 255).cpu().numpy().astype(np.uint8)
    return np.transpose(img_np, (1, 2, 0))  # CHW to HWC

def create_comparison_image(real_img, pred_img, env_name, mse_loss, is_best=True):
    """
    Create a side-by-side comparison image
    Args:
        real_img: Real image (H, W, C) numpy array
        pred_img: Predicted image (H, W, C) numpy array  
        env_name: Name of the environment
        mse_loss: MSE loss value
        is_best: Whether this is a best (low loss) or worst (high loss) example
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Real image (left)
    axes[0].imshow(real_img)
    axes[0].set_title('Real Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Predicted image (right)
    axes[1].imshow(pred_img)
    axes[1].set_title('Generated Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overall title with environment name and loss
    loss_type = "Best" if is_best else "Worst"
    fig.suptitle(f'{env_name} - {loss_type} Result\nMSE Loss: {mse_loss:.6f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_validation_environments():
    """创建4个不同游戏的验证环境"""
    env_managers = []
    
    # 为每个游戏创建单独的环境管理器
    for game in VALIDATION_GAMES:
        try:
            # 创建扩展的环境管理器，但只选择当前指定的游戏
            env_manager = ExtendedAtariEnvManager(num_games=4, num_envs=1, render_mode=None)
            
            # 手动设置当前环境为指定游戏
            if env_manager.envs:
                env_manager.envs[0].close()
                env_manager.envs[0] = gym.make(game, render_mode=None)
                env_manager.current_games[0] = game
                obs, _ = env_manager.envs[0].reset()
                env_manager.current_obs[0] = obs
                env_manager.episode_rewards[0] = 0
                
                env_managers.append(env_manager)
                print(f"成功创建验证环境: {game}")
            
        except Exception as e:
            print(f"创建游戏 {game} 的验证环境失败: {e}")
            continue
    
    return env_managers

def main():
    print(f"Using device: {DEVICE}")
    
    # Create save directory and log directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Clear existing images in the directory
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize evaluation log file
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    with open(LOG_FILE, 'w') as log_f:
        log_f.write("="*80 + "\n")
        log_f.write("模型验证详细日志\n")
        log_f.write("="*80 + "\n")
        log_f.write(f"验证开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"使用设备: {DEVICE}\n")
        log_f.write(f"验证步数: {NUM_VALIDATION_STEPS}\n")
        log_f.write(f"验证环境数量: {NUM_VALIDATION_ENVS}\n")
        log_f.write(f"模型路径: {MODEL_CHECKPOINT_PATH}\n")
        log_f.write("="*80 + "\n\n")
    
    # --- Initialize Environment Managers for 4 different games ---
    print("初始化4个不同游戏的验证环境...")
    env_managers = create_validation_environments()
    
    if len(env_managers) == 0:
        print("未能创建任何验证环境！")
        return
        
    print(f"成功创建 {len(env_managers)} 个验证环境: {[mg.current_games[0] for mg in env_managers]}")
    
    # Get action dimension from first environment
    action_dim = env_managers[0].envs[0].action_space.n
    print(f"Action dimension: {action_dim}")
    
    # --- Initialize and Load Model ---
    print("Initializing model...")
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
    
    # Load trained weights
    print(f"Loading model weights from {MODEL_CHECKPOINT_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
        print("Model weights loaded successfully!")
    except FileNotFoundError:
        print(f"\nWarning: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
        print("Using randomly initialized weights for demonstration...")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Using randomly initialized weights for demonstration...")
    
    model.eval()
    
    # --- Validation Data Collection ---
    print("Collecting validation data...")
    
    # Store validation data for each environment
    validation_data = defaultdict(list)  # {env_name: [(obs1, action, obs2, loss), ...]}
    
    steps_per_env = NUM_VALIDATION_STEPS // len(env_managers)
    
    # Log environment setup
    with open(LOG_FILE, 'a') as log_f:
        log_f.write("环境设置:\n")
        for i, env_manager in enumerate(env_managers):
            env_name = env_manager.current_games[0]
            log_f.write(f"环境 {i+1}: {env_name}\n")
        log_f.write(f"每个环境验证步数: {steps_per_env}\n")
        log_f.write("\n" + "-"*80 + "\n")
        log_f.write("详细验证日志:\n")
        log_f.write("-"*80 + "\n\n")
    
    for env_idx, env_manager in enumerate(env_managers):
        env_name = env_manager.current_games[0]
        print(f"Collecting data from {env_name}...")
        
        # Log start of environment validation
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"开始验证环境: {env_name}\n")
            log_f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write("-"*40 + "\n")
        
        env_losses = []  # Store losses for this environment
        
        for step in range(steps_per_env):
            # Sample action and step environment
            actions_list = env_manager.sample_actions()
            experiences = env_manager.step(actions_list)
            
            # Process the experience (we only have 1 env per manager)
            obs1, action, obs2, reward, done = experiences[0]
            
            # Preprocess observations for model
            obs1_batch = preprocess_observations([obs1], DEVICE)
            obs2_batch = preprocess_observations([obs2], DEVICE)
            
            # Prepare model inputs
            obs1_input = obs1_batch.unsqueeze(1)  # (1, 1, C, H, W)
            obs2_target = obs2_batch.unsqueeze(1)  # (1, 1, C, H, W)
            action_tensor = torch.tensor([action], dtype=torch.long, device=DEVICE).unsqueeze(1)  # (1, 1)
            
            # Get model prediction and calculate loss
            with torch.no_grad():
                _, predicted_obs2_raw = model(obs1_input, action_tensor)
                
                # Calculate MSE loss
                predicted_obs2_processed = predicted_obs2_raw.squeeze(1)  # (1, C, H, W)
                obs2_target_processed = obs2_target.squeeze(1)  # (1, C, H, W)
                
                # Crop prediction to match target size
                predicted_obs2_cropped = model._crop_output(predicted_obs2_processed)
                loss = F.mse_loss(predicted_obs2_cropped, obs2_target_processed).item()
                
                # Store validation data
                validation_data[env_name].append({
                    'obs1': obs1,
                    'obs2': obs2,
                    'predicted_obs2': predicted_obs2_cropped.squeeze(0),  # (C, H, W)
                    'action': action,
                    'loss': loss
                })
                
                env_losses.append(loss)
                
                # Log detailed step information
                with open(LOG_FILE, 'a') as log_f:
                    log_f.write(f"Step {step+1:4d}: Action={action:2d}, Reward={reward:6.2f}, Done={str(done):5s}, Loss={loss:.6f}")
                    if done:
                        log_f.write(" [EPISODE_END]")
                    log_f.write("\n")
            
            if (step + 1) % 50 == 0:
                print(f"  {env_name}: Collected {step + 1}/{steps_per_env} samples")
                
                # Log progress to file
                with open(LOG_FILE, 'a') as log_f:
                    avg_loss_so_far = np.mean(env_losses)
                    log_f.write(f">>> 进度: {step + 1}/{steps_per_env} samples, 当前平均损失: {avg_loss_so_far:.6f}\n")
        
        # Log environment summary
        with open(LOG_FILE, 'a') as log_f:
            final_avg_loss = np.mean(env_losses)
            min_loss = min(env_losses)
            max_loss = max(env_losses)
            std_loss = np.std(env_losses)
            
            log_f.write(f"\n{env_name} 验证完成:\n")
            log_f.write(f"  总样本数: {len(env_losses)}\n")
            log_f.write(f"  平均损失: {final_avg_loss:.6f}\n")
            log_f.write(f"  最小损失: {min_loss:.6f}\n")
            log_f.write(f"  最大损失: {max_loss:.6f}\n")
            log_f.write(f"  标准差: {std_loss:.6f}\n")
            log_f.write("="*40 + "\n\n")
    
    # --- Generate Comparison Images ---
    print("Generating comparison images...")
    
    # Log image generation start
    with open(LOG_FILE, 'a') as log_f:
        log_f.write("开始生成对比图像...\n")
        log_f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write("-"*60 + "\n")
    
    for env_name, data_list in validation_data.items():
        if len(data_list) == 0:
            continue
            
        print(f"Processing {env_name} with {len(data_list)} samples...")
        
        # Sort by loss (ascending for best, descending for worst)
        sorted_data = sorted(data_list, key=lambda x: x['loss'])
        
        # Get best 8 (lowest loss) and worst 8 (highest loss)
        best_8 = sorted_data[:8]
        worst_8 = sorted_data[-8:]
        
        # Log image generation details
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"处理环境: {env_name}\n")
            log_f.write(f"  总样本数: {len(data_list)}\n")
            log_f.write(f"  最佳8个样本损失范围: {best_8[0]['loss']:.6f} - {best_8[-1]['loss']:.6f}\n")
            log_f.write(f"  最差8个样本损失范围: {worst_8[0]['loss']:.6f} - {worst_8[-1]['loss']:.6f}\n")
        
        # Create environment-specific directory
        env_save_dir = os.path.join(SAVE_DIR, env_name.replace('-v4', ''))
        os.makedirs(env_save_dir, exist_ok=True)
        
        # Save best 8 results
        best_dir = os.path.join(env_save_dir, 'best_8')
        os.makedirs(best_dir, exist_ok=True)
        
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"  保存最佳8个结果:\n")
        
        for i, data in enumerate(best_8):
            real_img = data['obs2']
            pred_img = denormalize_image(data['predicted_obs2'])
            
            fig = create_comparison_image(
                real_img, pred_img, env_name, data['loss'], is_best=True
            )
            
            save_path = os.path.join(best_dir, f'best_{i+1}_loss_{data["loss"]:.6f}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            with open(LOG_FILE, 'a') as log_f:
                log_f.write(f"    best_{i+1}: loss={data['loss']:.6f}, action={data['action']}\n")
        
        # Save worst 8 results
        worst_dir = os.path.join(env_save_dir, 'worst_8')
        os.makedirs(worst_dir, exist_ok=True)
        
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"  保存最差8个结果:\n")
        
        for i, data in enumerate(worst_8):
            real_img = data['obs2']
            pred_img = denormalize_image(data['predicted_obs2'])
            
            fig = create_comparison_image(
                real_img, pred_img, env_name, data['loss'], is_best=False
            )
            
            save_path = os.path.join(worst_dir, f'worst_{i+1}_loss_{data["loss"]:.6f}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            with open(LOG_FILE, 'a') as log_f:
                log_f.write(f"    worst_{i+1}: loss={data['loss']:.6f}, action={data['action']}\n")
        
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"  {env_name} 图像保存完成\n")
            log_f.write("-"*40 + "\n")
        
        print(f"Saved comparison images for {env_name}")
        print(f"  Best loss: {best_8[0]['loss']:.6f}")
        print(f"  Worst loss: {worst_8[-1]['loss']:.6f}")
        print(f"  Average loss: {np.mean([d['loss'] for d in data_list]):.6f}")
    
    # --- Generate Summary Report ---
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    summary_file = os.path.join(SAVE_DIR, 'validation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Validation Summary\n")
        f.write("="*50 + "\n\n")
        
        for env_name, data_list in validation_data.items():
            if len(data_list) == 0:
                continue
                
            losses = [d['loss'] for d in data_list]
            
            summary = f"""
Environment: {env_name}
Samples: {len(data_list)}
Best Loss: {min(losses):.6f}
Worst Loss: {max(losses):.6f}
Average Loss: {np.mean(losses):.6f}
Std Loss: {np.std(losses):.6f}
"""
            print(summary)
            f.write(summary + "\n")
    
    print(f"Summary saved to: {summary_file}")
    print(f"All comparison images saved to: {SAVE_DIR}")
    
    # --- Cleanup ---
    for env_manager in env_managers:
        env_manager.close()
    print("验证完成！所有环境已关闭。")

if __name__ == '__main__':
    main()