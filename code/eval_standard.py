"""
通用模型验证脚本
================

该脚本提供通用的模型评估框架，只需修改MODEL_CONFIGS中的配置即可评估不同模型。
模型需要实现 evaluate_on_batch 和 forward 方法。

主要功能：
1. 支持多种模型的动态导入和初始化
2. 在4个不同的Atari游戏环境中收集验证数据
3. 调用模型的evaluate_on_batch方法计算损失
4. 生成三图对比可视化：输入图像 vs 真实图像 vs 生成图像
5. 保存每个环境下最佳和最差的8个结果

使用方法：
1. 在MODEL_CONFIGS中添加新模型配置
2. 修改CURRENT_MODEL指向要评估的模型
3. 运行脚本

模型要求：
- 实现 evaluate_on_batch(obs1_input, actions_input, obs2_target) -> float 方法
- 实现 forward(obs1_input, actions_input) -> (latent, predicted_obs2) 方法
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import shutil
from datetime import datetime
import importlib
import gymnasium as gym
import ale_py

# =============================================================================
# 模型配置区域 - 只需修改这个部分来切换不同模型
# =============================================================================

MODEL_CONFIGS = {
    # 原始 PPM attention 模型
    "ppm_attention": {
        "module_path": "model.PPM_attention",
        "class_name": "PredictiveRepModel", 
        "init_params": {
            "img_in_channels": 3,
            "img_out_channels": 3,
            "encoder_layers": [2, 2],
            "decoder_layers_config": [2, 2, 1],  # 注意这里参数名不同
            "target_img_h": 210,
            "target_img_w": 160,
            "action_dim": 18,
            "latent_dim": 256
        },
        "checkpoint_path": "Output/checkpoint/trained_models_attention/trained_models_attention/ppm_attention_step_20000.pth"
    },
    
    # 增强版 PPM attention 模型
    "enhanced_ppm_attention": {
        "module_path": "model.PPM_attention2", 
        "class_name": "EnhancedPredictiveRepModel",
        "init_params": {
            "img_in_channels": 3,
            "img_out_channels": 3,
            "encoder_layers": [2, 3, 4, 3],
            "decoder_layers": [2, 2, 2, 1],  # 注意参数名不同
            "target_img_h": 210,
            "target_img_w": 160,
            "action_dim": 18,
            "latent_dim": 256,
            "base_channels": 64,
            "num_attention_heads": 8,
            "use_skip_connections": True
        },
        "checkpoint_path": "Output/checkpoint/enhanced_trained_models_attention/enhanced_ppm_attention_step_10000.pth"
    },
    
    # 可以继续添加其他模型配置...
    # "your_new_model": {
    #     "module_path": "model.YourModel",
    #     "class_name": "YourModelClass",
    #     "init_params": {...},
    #     "checkpoint_path": "path/to/your/model.pth"
    # }
}

# 选择当前要评估的模型
CURRENT_MODEL = "enhanced_ppm_attention"  # 修改这里来切换模型

# =============================================================================
# 验证参数配置
# =============================================================================

NUM_VALIDATION_ENVS = 4
NUM_VALIDATION_STEPS = 100

LOG_DIR = "Output/Eval_res/Episode2"
IMG_SAVE_DIR = "Output/Eval_res/Episode2/Imgs"
LOG_FILE = "Output/Eval_res/Episode2/eval_log.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 验证游戏列表
VALIDATION_GAMES = [
    "Seaquest-v4",
    "Riverraid-v4", 
    "ChopperCommand-v4",
    "Breakout-v4"
]

NUM_SAMPLE_IMG = 4
steps_warm_up = 20  # 预热步数，确保模型有足够的输入数据
# =============================================================================
# 环境管理器
# =============================================================================
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 导入环境管理器
from env.AtariEnv_random import AtariEnvManager

class ExtendedAtariEnvManager(AtariEnvManager):
    """扩展的Atari环境管理器，添加第4个游戏用于验证"""
    def __init__(self, num_games, render_mode=None, num_envs=1):
        self.available_games = [
            "Seaquest-v4",
            "Riverraid-v4",
            "ChopperCommand-v4",
            "Breakout-v4"
        ]
        
        num_games = min(num_games, len(self.available_games))
        
        import numpy as np
        self.selected_games = np.random.choice(self.available_games, num_games, replace=False)
        print(f"验证游戏池中的游戏: {self.selected_games}")
        
        self.render_mode = render_mode
        self.effective_render_mode = [render_mode] + [None] * (num_envs - 1) if render_mode == 'human' else [None] * num_envs
        
        env_games = np.random.choice(self.selected_games, num_envs)
        print(f"尝试创建 {num_envs} 个验证环境实例")
        for i, game in enumerate(env_games):
            print(f"验证环境 {i+1} 初始游戏: {game}")

        self.envs = []
        self.current_games = []

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
        self.current_obs = [None] * self.num_envs
        self.episode_rewards = [0] * self.num_envs
        
        for i in range(self.num_envs):
            obs, _ = self.envs[i].reset()
            self.current_obs[i] = obs

# =============================================================================
# 通用模型加载器
# =============================================================================

def load_model_dynamically(model_config):
    """
    动态加载模型
    
    Args:
        model_config: 模型配置字典
        
    Returns:
        model: 初始化后的模型实例
    """
    try:
        # 动态导入模块
        module = importlib.import_module(model_config["module_path"])
        
        # 获取模型类
        model_class = getattr(module, model_config["class_name"])
        
        # 初始化模型
        model = model_class(**model_config["init_params"])
        
        print(f"成功加载模型: {model_config['class_name']}")
        print(f"模型参数: {model_config['init_params']}")
        
        return model
        
    except ImportError as e:
        print(f"模块导入失败: {model_config['module_path']}")
        print(f"错误: {e}")
        raise
    except AttributeError as e:
        print(f"模型类未找到: {model_config['class_name']}")
        print(f"错误: {e}")
        raise
    except Exception as e:
        print(f"模型初始化失败: {e}")
        raise

def load_model_weights(model, checkpoint_path, device):
    """
    加载模型权重，支持不同的保存格式
    
    Args:
        model: 模型实例
        checkpoint_path: 权重文件路径
        device: 设备
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            # 完整的checkpoint格式
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从完整checkpoint加载权重: {checkpoint_path}")
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"Checkpoint loss: {checkpoint['loss']}")
        else:
            # 仅权重格式
            model.load_state_dict(checkpoint)
            print(f"从权重文件加载: {checkpoint_path}")
            
        print("模型权重加载成功！")
        
    except FileNotFoundError:
        print(f"\n警告: 模型checkpoint未找到: {checkpoint_path}")
        print("使用随机初始化权重进行演示...")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        print("使用随机初始化权重进行演示...")

# =============================================================================
# 辅助函数
# =============================================================================

# 文件: eval_standard.py

def preprocess_observations(obs_list, device):
    """将观测列表转换为模型输入格式，并归一化到 [-1, 1]"""
    processed_obs = []
    for obs_hwc in obs_list:
        # 从 (H, W, C) 转换到 (C, H, W)
        obs_chw = np.transpose(obs_hwc, (2, 0, 1))
        processed_obs.append(obs_chw)
    
    # 转换为float32张量
    obs_tensor_bchw = torch.tensor(np.array(processed_obs), dtype=torch.float32, device=device)
    
    # 归一化到 [-1, 1] 范围，以匹配模型的Tanh输出
    normalized_tensor = (obs_tensor_bchw / 127.5) - 1.0
    
    return normalized_tensor

def denormalize_image(img_tensor):
    """
    将值域为 [-1, 1] 的tensor转换为可显示的 [0, 255] RGB图像格式。
    """
    # 1. 将值域从 [-1, 1] 映射到 [0, 1]
    img_tensor_norm = (img_tensor.cpu().detach() + 1) / 2.0
    
    # 2. 将值域从 [0, 1] 映射到 [0, 255]
    img_tensor_scaled = img_tensor_norm * 255.0
    
    # 3. 裁剪以确保值在 [0, 255] 范围内
    img_tensor_clamped = torch.clamp(img_tensor_scaled, 0, 255)
    
    # 4. 转换为numpy数组并更改数据类型
    img_np = img_tensor_clamped.numpy().astype(np.uint8)
    
    # 5. 将维度从 (C, H, W) 转换为 (H, W, C) 以便显示
    return np.transpose(img_np, (1, 2, 0))

def create_comparison_image(input_img, real_img, pred_img, env_name, mse_loss, is_best=True):
    """创建三图对比：输入图像、真实图像、生成图像"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 输入图像 (左侧)
    axes[0].imshow(input_img)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold', color='blue')
    axes[0].axis('off')
    
    # 真实图像 (中间)
    axes[1].imshow(real_img)
    axes[1].set_title('Real Image (Ground Truth)', fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # 生成图像 (右侧)
    axes[2].imshow(pred_img)
    axes[2].set_title('Generated Image (Prediction)', fontsize=14, fontweight='bold', color='red')
    axes[2].axis('off')
    
    loss_type = "Best" if is_best else "Worst"
    fig.suptitle(f'{env_name} - {loss_type} Result\nMSE Loss: {mse_loss:.6f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_validation_environments():
    """创建4个不同游戏的验证环境"""
    env_managers = []
    
    for game in VALIDATION_GAMES:
        try:
            env_manager = ExtendedAtariEnvManager(num_games=4, num_envs=1, render_mode=None)
            
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

# =============================================================================
# 主函数
# =============================================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"当前评估模型: {CURRENT_MODEL}")
    
    # 获取当前模型配置
    if CURRENT_MODEL not in MODEL_CONFIGS:
        raise ValueError(f"模型配置未找到: {CURRENT_MODEL}")
    
    model_config = MODEL_CONFIGS[CURRENT_MODEL]
    print(f"模型配置: {model_config}")
    
    # 创建保存目录
    os.makedirs(IMG_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 清除现有图像
    if os.path.exists(IMG_SAVE_DIR):
        shutil.rmtree(IMG_SAVE_DIR)
    os.makedirs(IMG_SAVE_DIR, exist_ok=True)
    
    # 初始化日志文件
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    with open(LOG_FILE, 'w') as log_f:
        log_f.write("="*80 + "\n")
        log_f.write("通用模型验证详细日志\n")
        log_f.write("="*80 + "\n")
        log_f.write(f"验证开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"使用设备: {DEVICE}\n")
        log_f.write(f"当前模型: {CURRENT_MODEL}\n")
        log_f.write(f"模型类: {model_config['class_name']}\n")
        log_f.write(f"验证步数: {NUM_VALIDATION_STEPS}\n")
        log_f.write(f"验证环境数量: {NUM_VALIDATION_ENVS}\n")
        log_f.write(f"模型路径: {model_config['checkpoint_path']}\n")
        log_f.write("="*80 + "\n\n")
    
    # 初始化验证环境
    print("初始化4个不同游戏的验证环境...")
    env_managers = create_validation_environments()
    
    if len(env_managers) == 0:
        print("未能创建任何验证环境！")
        return
        
    print(f"成功创建 {len(env_managers)} 个验证环境: {[mg.current_games[0] for mg in env_managers]}")
    
    # 获取动作维度
    action_dim = env_managers[0].envs[0].action_space.n
    print(f"Action dimension: {action_dim}")
    
    # 更新模型配置中的action_dim（如果需要）
    if 'action_dim' in model_config['init_params']:
        model_config['init_params']['action_dim'] = action_dim
    
    # 动态加载和初始化模型
    print("动态初始化模型...")
    model = load_model_dynamically(model_config)
    model = model.to(DEVICE)
    
    # 加载训练好的权重
    print(f"Loading model weights from {model_config['checkpoint_path']}...")
    load_model_weights(model, model_config['checkpoint_path'], DEVICE)
    
    model.eval()
    
    # 验证模型是否有必要的方法
    if not hasattr(model, 'evaluate_on_batch'):
        raise AttributeError(f"模型 {model_config['class_name']} 没有 evaluate_on_batch 方法")
    
    if not hasattr(model, 'forward'):
        raise AttributeError(f"模型 {model_config['class_name']} 没有 forward 方法")
    
    print("✅ 模型方法验证通过")
    
    # 验证数据收集
    print("Collecting validation data...")
    validation_data = defaultdict(list)
    steps_per_env = NUM_VALIDATION_STEPS // len(env_managers)
    
    # 记录环境设置
    with open(LOG_FILE, 'a') as log_f:
        log_f.write("环境设置:\n")
        for i, env_manager in enumerate(env_managers):
            env_name = env_manager.current_games[0]
            log_f.write(f"环境 {i+1}: {env_name}\n")
        log_f.write(f"每个环境验证步数: {steps_per_env}\n")
        log_f.write("\n" + "-"*80 + "\n")
        log_f.write("详细验证日志:\n")
        log_f.write("-"*80 + "\n\n")
    
    # 对每个环境进行验证
    for env_idx, env_manager in enumerate(env_managers):
        env_name = env_manager.current_games[0]
        print(f"Collecting data from {env_name}...")
        
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"开始验证环境: {env_name}\n")
            log_f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"环境预热部署: {steps_warm_up}\n")
            log_f.write("-"*40 + "\n")
        
        env_losses = []
        
        for step in range(steps_per_env + steps_warm_up):
            if step < steps_warm_up:
                # 预热阶段，随机采样动作
                actions_list = [env_manager.envs[0].action_space.sample() for _ in range(env_manager.num_envs)]
                experiences = env_manager.step(actions_list)
                if step == steps_warm_up - 1:
                    print(f"{env_name}: 预热结束，开始验证...")
                    with open(LOG_FILE, 'a') as log_f:
                        log_f.write(f"{env_name}: 预热结束，开始验证...\n")
            else:# 采样动作并执行
                actions_list = env_manager.sample_actions()
                experiences = env_manager.step(actions_list)
                
                obs1, action, obs2, reward, done = experiences[0]
                
                # 预处理观测
                obs1_batch = preprocess_observations([obs1], DEVICE)
                obs2_batch = preprocess_observations([obs2], DEVICE)
                
                # 准备模型输入格式
                obs1_input = obs1_batch.unsqueeze(1)  # (1, 1, C, H, W)
                obs2_target = obs2_batch.unsqueeze(1)  # (1, 1, C, H, W)
                actions_input = torch.tensor([action], dtype=torch.long, device=DEVICE).unsqueeze(1)  # (1, 1)
                
                # 使用模型的evaluate_on_batch方法计算损失
                loss = model.evaluate_on_batch(obs1_input, actions_input, obs2_target)
                
                # 获取预测图像用于可视化
                with torch.no_grad():
                    _, predicted_obs2_raw = model(obs1_input, actions_input)
                    predicted_obs2_processed = predicted_obs2_raw.squeeze(1)  # (1, C, H, W)
                
                # 存储验证数据
                validation_data[env_name].append({
                    'obs1': obs1,  # 输入图像
                    'obs2': obs2,  # 真实下一帧图像
                    'predicted_obs2': predicted_obs2_processed.squeeze(0),  # (C, H, W)
                    'action': action,
                    'loss': loss
                })
                
                env_losses.append(loss)
                
                # 记录详细步骤信息
                with open(LOG_FILE, 'a') as log_f:
                    log_f.write(f"Step {step+1:4d}: Action={action:2d}, Reward={reward:6.2f}, Done={str(done):5s}, Loss={loss:.6f}")
                    if done:
                        log_f.write(" [EPISODE_END]")
                    log_f.write("\n")
                
                if (step + 1) % 50 == 0:
                    print(f"  {env_name}: Collected {step + 1}/{steps_per_env} samples")
                    
                    with open(LOG_FILE, 'a') as log_f:
                        avg_loss_so_far = np.mean(env_losses)
                        log_f.write(f">>> 进度: {step + 1}/{steps_per_env} samples, 当前平均损失: {avg_loss_so_far:.6f}\n")
            
        # 记录环境总结
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
    
    # 生成对比图像
    print("Generating comparison images...")
    
    with open(LOG_FILE, 'a') as log_f:
        log_f.write("开始生成对比图像...\n")
        log_f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write("-"*60 + "\n")
    
    for env_name, data_list in validation_data.items():
        if len(data_list) == 0:
            continue
            
        print(f"Processing {env_name} with {len(data_list)} samples...")
        
        # 按损失排序
        sorted_data = sorted(data_list, key=lambda x: x['loss'])

        best_k = sorted_data[:NUM_SAMPLE_IMG]
        worst_k = sorted_data[-NUM_SAMPLE_IMG:]
        
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"处理环境: {env_name}\n")
            log_f.write(f"  总样本数: {len(data_list)}\n")
            log_f.write(f"  最佳{NUM_SAMPLE_IMG}个样本损失范围: {best_k[0]['loss']:.6f} - {best_k[-1]['loss']:.6f}\n")
            log_f.write(f"  最差{NUM_SAMPLE_IMG}个样本损失范围: {worst_k[0]['loss']:.6f} - {worst_k[-1]['loss']:.6f}\n")
        
        # 创建环境特定目录
        env_save_dir = os.path.join(IMG_SAVE_DIR, env_name.replace('-v4', ''))
        os.makedirs(env_save_dir, exist_ok=True)
        
        # 保存最佳NUM_SAMPLE_IMG个结果
        best_dir = os.path.join(env_save_dir, f'best_{NUM_SAMPLE_IMG}')
        os.makedirs(best_dir, exist_ok=True)
        
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"  保存最佳{NUM_SAMPLE_IMG}个结果:\n")
        
        for i, data in enumerate(best_k):
            input_img = data['obs1']  # 输入图像
            real_img = data['obs2']   # 真实图像
            pred_img = denormalize_image(data['predicted_obs2'])  # 生成图像
            
            fig = create_comparison_image(input_img, real_img, pred_img, env_name, data['loss'], is_best=True)
            
            save_path = os.path.join(best_dir, f'best_{i+1}_loss_{data["loss"]:.6f}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            with open(LOG_FILE, 'a') as log_f:
                log_f.write(f"    best_{i+1}: loss={data['loss']:.6f}, action={data['action']}\n")
        
        # 保存最差NUM_SAMPLE_IMG个结果
        worst_dir = os.path.join(env_save_dir, f'worst_{NUM_SAMPLE_IMG}')
        os.makedirs(worst_dir, exist_ok=True)
        
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"  保存最差{NUM_SAMPLE_IMG}个结果:\n")
        
        for i, data in enumerate(worst_k):
            input_img = data['obs1']  # 输入图像  
            real_img = data['obs2']   # 真实图像
            pred_img = denormalize_image(data['predicted_obs2'])  # 生成图像
            
            fig = create_comparison_image(input_img, real_img, pred_img, env_name, data['loss'], is_best=False)
            
            save_path = os.path.join(worst_dir, f'worst_{i+1}_loss_{data["loss"]:.6f}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            with open(LOG_FILE, 'a') as log_f:
                log_f.write(f"    worst_{i+1}: loss={data['loss']:.6f}, action={data['action']}\n")
        
        with open(LOG_FILE, 'a') as log_f:
            log_f.write(f"  {env_name} 图像保存完成\n")
            log_f.write("-"*40 + "\n")
        
        print(f"Saved comparison images for {env_name}")
        print(f"  Best loss: {best_k[0]['loss']:.6f}")
        print(f"  Worst loss: {worst_k[-1]['loss']:.6f}")
        print(f"  Average loss: {np.mean([d['loss'] for d in data_list]):.6f}")
    
    # 生成总结报告
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    summary_file = os.path.join(IMG_SAVE_DIR, 'validation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Validation Summary - {CURRENT_MODEL}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {model_config['class_name']}\n")
        f.write(f"Checkpoint: {model_config['checkpoint_path']}\n\n")
        
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
    print(f"All comparison images saved to: {IMG_SAVE_DIR}")
    
    # 清理环境
    for env_manager in env_managers:
        env_manager.close()
    print("验证完成！所有环境已关闭。")

if __name__ == '__main__':
    main()