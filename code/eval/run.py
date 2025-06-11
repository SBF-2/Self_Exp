# main_runner.py
import torch
import torch.optim as optim
import numpy as np
import time # 用于计时

# 从其他模块导入
from env.AtariEnv import AtariEnvManager
from model.PPM import PredictiveRepModel
import wandb

# --- 🔴 配置参数 ---
GAME_NAME = "Seaquest"          # Atari 游戏名称 ("RiverRaid-v4", "ChopperCommand" 等)
NUM_ENVS = 4                    # 并行环境数量 (用于数据收集)
BATCH_SIZE = 32                 # 模型训练的批次大小
LEARNING_RATE = 1e-4            # 学习率
NUM_TRAINING_STEPS = 1000      # 总训练步数
EVAL_INTERVAL = 500             # 每隔多少步进行一次评估
PRINT_INTERVAL = 20             # 每隔多少步打印一次训练损失
SAVE_INTERVAL = 500            # 每隔多少步保存一次模型

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# Atari 观测属性 (来自环境)
# 这些值在预处理和模型初始化时使用
RAW_IMG_HEIGHT = 210
RAW_IMG_WIDTH = 160
RAW_IMG_CHANNELS = 3

# 模型参数
MODEL_ENCODER_LAYERS = [2, 2]   # 编码器中每个 DownBlock 的层数
# 解码器有3个UpBlock, 这里配置每个UpBlock的层数
MODEL_DECODER_LAYERS_CONFIG = [MODEL_ENCODER_LAYERS[1], MODEL_ENCODER_LAYERS[0], 1] # 例如 [2,2,1]
MODEL_OUTPUT_CHANNELS = RAW_IMG_CHANNELS # 模型输出通道数 (通常与输入一致)

# --- 数据预处理函数 ---
def preprocess_observations_actions(obs_batch_np, actions_batch_np, device):
    """
    将 NumPy 格式的观测和动作批次转换为 PyTorch 张量，并进行预处理。
    参数:
        obs_batch_np (np.ndarray): 形状 (N, H, W, C) 的观测数据。
        actions_batch_np (np.ndarray): 形状 (N,) 的动作数据。
        device: PyTorch 设备。
    返回:
        obs_tensor (torch.Tensor): 形状 (N, 1, C, H, W)
        actions_tensor (torch.Tensor): 形状 (N, 1)
    """
    # 1. 处理观测数据
    #    从 (N, H, W, C) 转为 (N, C, H, W)，归一化到 [0, 1]
    obs_tensor_n_c_h_w = torch.tensor(obs_batch_np, dtype=torch.float32, device=device)
    obs_tensor_n_c_h_w = obs_tensor_n_c_h_w.permute(0, 3, 1, 2) / 255.0
    
    #    增加序列长度维度 L=1: (N, C, H, W) -> (N, 1, C, H, W)
    obs_tensor_n_l_c_h_w = obs_tensor_n_c_h_w.unsqueeze(1)

    # 2. 处理动作数据
    actions_tensor_n = torch.tensor(actions_batch_np, dtype=torch.long, device=device)
    #    增加序列长度维度 L=1: (N,) -> (N, 1)
    actions_tensor_n_l = actions_tensor_n.unsqueeze(1)
    
    return obs_tensor_n_l_c_h_w, actions_tensor_n_l

# --- 主训练与评估流程 ---
def main():
    print("--- 初始化环境和模型 ---")
    # 1. 初始化环境管理器
    #    训练时通常不进行渲染 (render_mode=None)
    env_manager = AtariEnvManager(GAME_NAME, num_envs=NUM_ENVS, render_mode=None)

    # 2. 初始化模型
    model = PredictiveRepModel(
        img_in_channels=RAW_IMG_CHANNELS,
        img_out_channels=MODEL_OUTPUT_CHANNELS,
        encoder_layers=MODEL_ENCODER_LAYERS,
        decoder_layers_config=MODEL_DECODER_LAYERS_CONFIG,
        target_img_h=RAW_IMG_HEIGHT, # 目标图像高度 (用于裁剪模型输出)
        target_img_w=RAW_IMG_WIDTH,  # 目标图像宽度
    ).to(DEVICE)

    # 3. 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"--- 开始在 {DEVICE} 上训练 {NUM_TRAINING_STEPS} 步 ---")
    start_time = time.time()

    # 训练循环
    for step in range(1, NUM_TRAINING_STEPS + 1):
        # --- 收集一个批次的数据 ---
        batch_obs1_list, batch_actions_list, batch_obs2_list = [], [], []
        
        # 循环直到收集到足够的 BATCH_SIZE 个经验
        while len(batch_obs1_list) < BATCH_SIZE:
            # 从所有并行环境中采样动作
            actions_to_take_in_envs = env_manager.sample_actions()
            
            # 在所有环境中执行一步，获取经验元组列表
            # experiences_from_envs: [(obs1, action, obs2, reward, done), ...]
            experiences_from_envs = env_manager.step(actions_to_take_in_envs)
            
            for obs1_np, action_int, obs2_np, _, _ in experiences_from_envs:
                if len(batch_obs1_list) < BATCH_SIZE: # 确保不超过 BATCH_SIZE
                    batch_obs1_list.append(obs1_np)       # HWC, uint8
                    batch_actions_list.append(action_int) # int
                    batch_obs2_list.append(obs2_np)       # HWC, uint8
                else:
                    break # 当前批次数据已满

        # 将列表转换为 NumPy 数组
        obs1_np_batch = np.array(batch_obs1_list)
        actions_np_batch = np.array(batch_actions_list)
        obs2_np_batch = np.array(batch_obs2_list)

        # --- 数据预处理 ---
        # obs1_input: (BATCH_SIZE, 1, C, H, W), float, normalized
        # actions_input: (BATCH_SIZE, 1), long
        obs1_input, actions_input = preprocess_observations_actions(obs1_np_batch, actions_np_batch, DEVICE)
        
        # obs2_target 也需要同样的处理 (除了动作)
        obs2_target, _ = preprocess_observations_actions(obs2_np_batch, np.zeros_like(actions_np_batch), DEVICE) # 动作部分不使用

        # --- 模型训练 ---
        train_loss = model.train_on_batch(obs1_input, actions_input, obs2_target, optimizer)

        # --- 打印训练信息 ---
        if step % PRINT_INTERVAL == 0:
            elapsed_time = time.time() - start_time
            print(f"训练步 [{step}/{NUM_TRAINING_STEPS}], "
                  f"损失: {train_loss:.6f}, "
                  f"耗时: {elapsed_time:.2f}s")
            start_time = time.time() # 重置计时器

        # --- 定期评估 ---
        if step % EVAL_INTERVAL == 0:
            # 收集评估数据 (与训练数据收集方式相同，但通常来自不同的数据集或状态)
            # 这里为了简化，我们使用与训练时类似的方式收集新的一批数据进行评估
            eval_obs1_list, eval_actions_list, eval_obs2_list = [], [], []
            while len(eval_obs1_list) < BATCH_SIZE: # 使用相同的 BATCH_SIZE 进行评估
                actions_to_take_in_envs = env_manager.sample_actions()
                experiences_from_envs = env_manager.step(actions_to_take_in_envs)
                for o1, a, o2, _, _ in experiences_from_envs:
                    if len(eval_obs1_list) < BATCH_SIZE:
                        eval_obs1_list.append(o1)
                        eval_actions_list.append(a)
                        eval_obs2_list.append(o2)
                    else: break
            
            eval_o1_np = np.array(eval_obs1_list)
            eval_a_np = np.array(eval_actions_list)
            eval_o2_np = np.array(eval_obs2_list)

            eval_obs1_input, eval_actions_input = preprocess_observations_actions(eval_o1_np, eval_a_np, DEVICE)
            eval_obs2_target, _ = preprocess_observations_actions(eval_o2_np, np.zeros_like(eval_a_np), DEVICE)
            
            eval_loss = model.evaluate_on_batch(eval_obs1_input, eval_actions_input, eval_obs2_target)
            print(f"--- 评估步 [{step}/{NUM_TRAINING_STEPS}], 评估损失: {eval_loss:.6f} ---")
        
        # --- 定期保存模型 ---
        if step % SAVE_INTERVAL == 0:
            model_save_path = f"{GAME_NAME.lower()}_predcoder_step_{step}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到: {model_save_path}")

    print("--- 训练完成 ---")
    final_model_path = f"{GAME_NAME.lower()}_predcoder_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")

    env_manager.close() # 关闭环境
    # --- Wandb 初始化和配置 ---

    # 初始化 wandb
    wandb.init(
        project=f"predcoder-{GAME_NAME}",
        config={
            "game": GAME_NAME,
            "num_envs": NUM_ENVS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_training_steps": NUM_TRAINING_STEPS,
            "encoder_layers": MODEL_ENCODER_LAYERS,
            "decoder_layers": MODEL_DECODER_LAYERS_CONFIG,
        }
    )

    def log_metrics(step, train_loss=None, eval_loss=None):
        """记录训练指标到 wandb"""
        metrics = {}
        if train_loss is not None:
            metrics["train_loss"] = train_loss
        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss
        wandb.log(metrics, step=step)

    # 在主函数结束时关闭 wandb
    def cleanup():
        """清理函数"""
        wandb.finish()

if __name__ == '__main__':
    main()