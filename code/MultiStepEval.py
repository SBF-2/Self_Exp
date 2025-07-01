import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime

# 导入你的模型和环境
from model.PRM_JE import PRM_JE
from env.AtariEnv import AtariEnvManager


class PRM_Inference:
    """PRM_JE模型推理验证类"""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化推理器
        Args:
            model_path: 训练好的模型权重路径
            device: 计算设备
        """
        self.device = device
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 推理配置
        self.validation_steps = [1, 11, 21, 31]  # 进行损失计算的步数
        self.max_steps = 31  # 最大推理步数
        self.num_episodes = 100  # 每个环境的验证回合数
        
        # 存储结果
        self.results = {
            'losses': [],
            'cosine_similarities': [],
            'step_wise_losses': {step: [] for step in self.validation_steps}
        }
        
    def _load_model(self, model_path: str) -> PRM_JE:
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
        
        # 创建模型实例（使用与训练时相同的配置）
        model = PRM_JE(
            img_in_channels=3,
            encoder_layers=[2, 3, 4, 3],
            action_dim=19,  # 0-18共19个动作
            latent_dim=256,
            base_channels=64,
            num_attention_heads=8,
            transformer_layers=3,
            loss_weight=1.0,
            dropout=0.1
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        
        print("✅ 模型加载成功")
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理Atari图像
        Args:
            image: numpy数组，形状为(210, 160, 3)
        Returns:
            tensor: 预处理后的图像张量，形状为(1, 3, 210, 160)
        """
        # 确保图像在正确的范围内
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # 转换为torch tensor并调整维度顺序
        # (H, W, C) -> (C, H, W) -> (1, C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def recursive_inference(self, initial_image: torch.Tensor, actions: List[int]) -> List[torch.Tensor]:
        """
        递归推理：使用初始图像和动作序列进行多步预测
        Args:
            initial_image: 初始图像 (1, 3, 210, 160)
            actions: 动作序列
        Returns:
            predicted_features: 每一步的预测特征列表
        """
        predicted_features = []
        
        with torch.no_grad():
            # 获取初始图像的编码特征
            current_features = self.model.encoder(initial_image)  # (1, 256)
            
            for step, action in enumerate(actions):
                # 将动作转换为tensor
                action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
                
                # 使用predictive model预测下一步特征
                predicted_next_features = self.model.predictive_model(current_features, action_tensor)
                predicted_features.append(predicted_next_features)
                
                # 更新当前特征为预测特征（用于下一步预测）
                current_features = predicted_next_features
        
        return predicted_features
    
    def validate_episode(self, env_manager: AtariEnvManager, env_idx: int) -> Dict:
        """
        验证单个回合 - 确保每个episode独立进行完整的31步验证
        Args:
            env_manager: 环境管理器
            env_idx: 环境索引
        Returns:
            episode_results: 回合结果字典
        """
        episode_losses = {step: [] for step in self.validation_steps}
        episode_cosine_sims = {step: [] for step in self.validation_steps}
        # 添加真值间损失计算
        episode_ground_truth_losses = {step: [] for step in self.validation_steps}
        episode_ground_truth_cosine_sims = {step: [] for step in self.validation_steps}
        
        # 重置环境以确保干净的开始状态
        initial_obs, _ = env_manager.envs[env_idx].reset()
        env_manager.current_obs[env_idx] = initial_obs
        
        # 生成随机动作序列
        actions = [env_manager.envs[env_idx].action_space.sample() for _ in range(self.max_steps)]
        
        # 获取初始观测图像
        initial_image = self.preprocess_image(initial_obs)
        
        # 获取初始图像的encoder特征（用于真值比较）
        with torch.no_grad():
            initial_encoder_features = self.model.encoder(initial_image)  # (1, 256)
        
        # 递归推理 - 基于初始图像和动作序列预测所有31步的特征
        predicted_features = self.recursive_inference(initial_image, actions)
        
        # 在真实环境中执行相同的动作序列，并在指定步数进行验证
        current_obs = initial_obs
        step_count = 0
        episode_terminated = False
        
        for step_idx in range(self.max_steps):
            if episode_terminated:
                # 如果episode提前结束，就不能继续验证了
                break
                
            # 执行动作
            action = actions[step_idx]
            obs2, reward, terminated, truncated, info = env_manager.envs[env_idx].step(action)
            done = terminated or truncated
            
            current_obs = obs2
            step_count += 1
            
            # 在验证步数处计算损失
            if step_count in self.validation_steps:
                # 获取真实图像的编码特征
                real_image = self.preprocess_image(current_obs)
                with torch.no_grad():
                    real_features = self.model.encoder(real_image)  # (1, 256)
                
                # 获取对应的预测特征
                pred_features = predicted_features[step_idx]  # (1, 256)
                
                # 1. 计算预测特征与真实特征的损失（原有功能）
                pred_norm = F.normalize(pred_features, p=2, dim=1)
                real_norm = F.normalize(real_features, p=2, dim=1)
                cosine_sim = (pred_norm * real_norm).sum(dim=1).item()
                loss = 1.0 - cosine_sim
                
                episode_losses[step_count].append(loss)
                episode_cosine_sims[step_count].append(cosine_sim)
                
                # 2. 计算初始图像encoder与当前真实图像encoder的损失（新功能）
                initial_norm = F.normalize(initial_encoder_features, p=2, dim=1)
                ground_truth_cosine_sim = (initial_norm * real_norm).sum(dim=1).item()
                ground_truth_loss = 1.0 - ground_truth_cosine_sim
                
                episode_ground_truth_losses[step_count].append(ground_truth_loss)
                episode_ground_truth_cosine_sims[step_count].append(ground_truth_cosine_sim)
            
            # 检查环境是否结束
            if done:
                episode_terminated = True
                # 不在这里重置，让每个episode完整独立
        
        # episode结束后的清理工作会在下一个episode开始时进行
        
        return {
            'losses': episode_losses,
            'cosine_similarities': episode_cosine_sims,
            'ground_truth_losses': episode_ground_truth_losses,
            'ground_truth_cosine_similarities': episode_ground_truth_cosine_sims,
            'steps_completed': step_count,
            'episode_completed': step_count >= self.max_steps or episode_terminated
        }
    
    def run_validation(self, num_games: int = 4, num_envs: int = 4) -> Dict:
        """
        运行完整验证 - 确保每个环境对应一个特定游戏
        Args:
            num_games: 游戏数量（将被忽略，固定使用4种游戏）
            num_envs: 环境数量（将被忽略，固定使用4个环境）
        Returns:
            validation_results: 验证结果
        """
        # 固定使用4种不同的游戏，每种一个环境
        specified_games = [
            "Seaquest-v4",
            "Riverraid-v4", 
            "ChopperCommand-v4",
            "SpaceInvaders-v4"  # 添加第四种游戏
        ]
        
        print(f"开始验证 - 4种指定游戏, 每游戏{self.num_episodes}回合")
        print(f"游戏列表: {specified_games}")
        
        # 创建指定游戏的环境管理器
        env_manager = self._create_specified_envs(specified_games)
        
        # 获取环境名称
        env_names = specified_games
        
        all_results = {
            'step_wise_losses': {step: [] for step in self.validation_steps},
            'step_wise_cosine_sims': {step: [] for step in self.validation_steps},
            'step_wise_ground_truth_losses': {step: [] for step in self.validation_steps},
            'step_wise_ground_truth_cosine_sims': {step: [] for step in self.validation_steps},
            'env_results': [],
            'env_names': env_names,
            'detailed_data': []  # 用于CSV导出的详细数据
        }
        
        try:
            # 对每个环境进行验证
            for env_idx in range(len(env_names)):
                env_name = env_names[env_idx]
                print(f"\n验证环境 {env_idx + 1}/4 ({env_name})")
                env_results = {
                    'env_name': env_name,
                    'losses': {step: [] for step in self.validation_steps},
                    'cosine_similarities': {step: [] for step in self.validation_steps},
                    'ground_truth_losses': {step: [] for step in self.validation_steps},
                    'ground_truth_cosine_similarities': {step: [] for step in self.validation_steps}
                }
                
                # 进行多个回合验证 - 确保收集到足够的完整episode
                completed_episodes = 0
                attempt_count = 0
                max_attempts = self.num_episodes * 2  # 最多尝试2倍的次数
                
                # 创建进度条
                pbar = tqdm(total=self.num_episodes, desc=f"环境{env_idx+1} ({env_name})")
                
                while completed_episodes < self.num_episodes and attempt_count < max_attempts:
                    episode_result = self.validate_episode(env_manager, env_idx)
                    attempt_count += 1
                    
                    # 只有当episode成功完成验证时才记录结果
                    if episode_result['episode_completed']:
                        # 累积结果
                        for step in self.validation_steps:
                            if episode_result['losses'][step]:
                                loss_val = episode_result['losses'][step][0]
                                cosine_val = episode_result['cosine_similarities'][step][0]
                                gt_loss_val = episode_result['ground_truth_losses'][step][0]
                                gt_cosine_val = episode_result['ground_truth_cosine_similarities'][step][0]
                                
                                # 预测损失
                                env_results['losses'][step].append(loss_val)
                                env_results['cosine_similarities'][step].append(cosine_val)
                                all_results['step_wise_losses'][step].append(loss_val)
                                all_results['step_wise_cosine_sims'][step].append(cosine_val)
                                
                                # 真值损失
                                env_results['ground_truth_losses'][step].append(gt_loss_val)
                                env_results['ground_truth_cosine_similarities'][step].append(gt_cosine_val)
                                all_results['step_wise_ground_truth_losses'][step].append(gt_loss_val)
                                all_results['step_wise_ground_truth_cosine_sims'][step].append(gt_cosine_val)
                                
                                # 添加到详细数据中（用于CSV）
                                all_results['detailed_data'].append({
                                    'env_name': env_name,
                                    'env_index': env_idx,
                                    'episode': completed_episodes,  # 使用完成的episode计数
                                    'validation_step': step,
                                    'prediction_loss': loss_val,
                                    'prediction_cosine_similarity': cosine_val,
                                    'ground_truth_loss': gt_loss_val,
                                    'ground_truth_cosine_similarity': gt_cosine_val,
                                    'steps_completed': episode_result['steps_completed'],
                                    'attempt_number': attempt_count
                                })
                        
                        completed_episodes += 1
                        pbar.update(1)  # 更新进度条
                        
                    else:
                        # 如果episode没有完成31步，偶尔输出调试信息
                        if attempt_count % 20 == 0:  # 每20次尝试输出一次
                            pbar.set_postfix({
                                'attempts': attempt_count, 
                                'completed': completed_episodes,
                                'last_steps': episode_result['steps_completed']
                            })
                
                pbar.close()  # 关闭进度条
                
                if completed_episodes < self.num_episodes:
                    print(f"  警告: 环境{env_idx+1}只完成了{completed_episodes}/{self.num_episodes}个完整episode")
                
                all_results['env_results'].append(env_results)
                
                # 打印环境统计
                print(f"环境 {env_idx + 1} ({env_name}) 完成:")
                for step in self.validation_steps:
                    if env_results['losses'][step]:
                        avg_pred_loss = np.mean(env_results['losses'][step])
                        avg_pred_cosine = np.mean(env_results['cosine_similarities'][step])
                        avg_gt_loss = np.mean(env_results['ground_truth_losses'][step])
                        avg_gt_cosine = np.mean(env_results['ground_truth_cosine_similarities'][step])
                        print(f"  步数 {step:2d}: Pred Loss={avg_pred_loss:.6f}, Pred Cosine={avg_pred_cosine:.4f}")
                        print(f"          GT Loss={avg_gt_loss:.6f}, GT Cosine={avg_gt_cosine:.4f}")
        
        finally:
            env_manager.close()
        
        return all_results
    
    def _create_specified_envs(self, game_names: List[str]):
        """
        创建指定游戏的环境管理器
        Args:
            game_names: 游戏名称列表
        Returns:
            环境管理器
        """
        import gymnasium as gym
        import ale_py
        
        gym.register_envs(ale_py)
        
        class SpecifiedAtariEnvManager:
            def __init__(self, game_names):
                self.envs = []
                self.current_obs = []
                self.episode_rewards = []
                
                print(f"创建指定的{len(game_names)}个环境:")
                for i, game_name in enumerate(game_names):
                    try:
                        env = gym.make(game_name, render_mode=None)
                        self.envs.append(env)
                        
                        # 初始化观测
                        obs, _ = env.reset()
                        self.current_obs.append(obs)
                        self.episode_rewards.append(0)
                        
                        print(f"  环境 {i+1}: {game_name} - 动作空间: {env.action_space}")
                        
                    except Exception as e:
                        print(f"  创建环境 {game_name} 失败: {e}")
                        raise
                
                if not self.envs:
                    raise RuntimeError("未能成功创建任何环境实例。")
                
                self.num_envs = len(self.envs)
            
            def close(self):
                """关闭所有环境"""
                for env in self.envs:
                    env.close()
                print(f"{self.num_envs} 个指定环境已关闭。")
        
        return SpecifiedAtariEnvManager(game_names)
    
    def save_results(self, results: Dict, output_dir: str = "inference_results"):
        """保存验证结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存详细CSV结果
        self.save_csv_results(results, output_dir, timestamp)
        
        # 2. 保存统计汇总CSV
        self.save_summary_csv(results, output_dir, timestamp)
        
        # 3. 保存详细JSON结果
        results_file = os.path.join(output_dir, f"validation_results_{timestamp}.json")
        json_results = {}
        for key, value in results.items():
            if key == 'detailed_data':
                continue  # 跳过详细数据，已保存到CSV
            elif isinstance(value, dict):
                json_results[key] = {k: [float(x) for x in v] if isinstance(v, list) else v 
                                   for k, v in value.items()}
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"JSON结果已保存至: {results_file}")
        
        # 4. 生成统计报告
        self.generate_report(results, output_dir, timestamp)
        
        # 5. 生成可视化图表
        self.plot_results(results, output_dir, timestamp)
    
    def save_csv_results(self, results: Dict, output_dir: str, timestamp: str):
        """保存详细的CSV结果"""
        # 创建DataFrame
        df = pd.DataFrame(results['detailed_data'])
        
        # 保存详细数据CSV
        csv_file = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"详细CSV结果已保存至: {csv_file}")
        
        # 创建按环境和步数分组的透视表 - 预测损失
        pivot_pred_loss = df.pivot_table(
            values='prediction_loss', 
            index=['env_name', 'episode'], 
            columns='validation_step', 
            aggfunc='mean'
        )
        pivot_pred_cosine = df.pivot_table(
            values='prediction_cosine_similarity', 
            index=['env_name', 'episode'], 
            columns='validation_step', 
            aggfunc='mean'
        )
        
        # 创建按环境和步数分组的透视表 - 真值损失
        pivot_gt_loss = df.pivot_table(
            values='ground_truth_loss', 
            index=['env_name', 'episode'], 
            columns='validation_step', 
            aggfunc='mean'
        )
        pivot_gt_cosine = df.pivot_table(
            values='ground_truth_cosine_similarity', 
            index=['env_name', 'episode'], 
            columns='validation_step', 
            aggfunc='mean'
        )
        
        # 保存预测损失透视表
        pivot_pred_loss_file = os.path.join(output_dir, f"prediction_loss_by_env_step_{timestamp}.csv")
        pivot_pred_loss.to_csv(pivot_pred_loss_file)
        print(f"预测损失透视表已保存至: {pivot_pred_loss_file}")
        
        # 保存预测余弦相似度透视表
        pivot_pred_cosine_file = os.path.join(output_dir, f"prediction_cosine_by_env_step_{timestamp}.csv")
        pivot_pred_cosine.to_csv(pivot_pred_cosine_file)
        print(f"预测余弦相似度透视表已保存至: {pivot_pred_cosine_file}")
        
        # 保存真值损失透视表
        pivot_gt_loss_file = os.path.join(output_dir, f"ground_truth_loss_by_env_step_{timestamp}.csv")
        pivot_gt_loss.to_csv(pivot_gt_loss_file)
        print(f"真值损失透视表已保存至: {pivot_gt_loss_file}")
        
        # 保存真值余弦相似度透视表
        pivot_gt_cosine_file = os.path.join(output_dir, f"ground_truth_cosine_by_env_step_{timestamp}.csv")
        pivot_gt_cosine.to_csv(pivot_gt_cosine_file)
        print(f"真值余弦相似度透视表已保存至: {pivot_gt_cosine_file}")
    
    def save_summary_csv(self, results: Dict, output_dir: str, timestamp: str):
        """保存统计汇总CSV"""
        summary_data = []
        
        # 按环境汇总
        for env_result in results['env_results']:
            env_name = env_result['env_name']
            for step in self.validation_steps:
                pred_losses = env_result['losses'][step]
                pred_cosines = env_result['cosine_similarities'][step]
                gt_losses = env_result['ground_truth_losses'][step]
                gt_cosines = env_result['ground_truth_cosine_similarities'][step]
                
                if pred_losses:
                    summary_data.append({
                        'env_name': env_name,
                        'validation_step': step,
                        # 预测损失统计
                        'mean_prediction_loss': np.mean(pred_losses),
                        'std_prediction_loss': np.std(pred_losses),
                        'min_prediction_loss': np.min(pred_losses),
                        'max_prediction_loss': np.max(pred_losses),
                        'mean_prediction_cosine_similarity': np.mean(pred_cosines),
                        'std_prediction_cosine_similarity': np.std(pred_cosines),
                        'min_prediction_cosine_similarity': np.min(pred_cosines),
                        'max_prediction_cosine_similarity': np.max(pred_cosines),
                        # 真值损失统计
                        'mean_ground_truth_loss': np.mean(gt_losses),
                        'std_ground_truth_loss': np.std(gt_losses),
                        'min_ground_truth_loss': np.min(gt_losses),
                        'max_ground_truth_loss': np.max(gt_losses),
                        'mean_ground_truth_cosine_similarity': np.mean(gt_cosines),
                        'std_ground_truth_cosine_similarity': np.std(gt_cosines),
                        'min_ground_truth_cosine_similarity': np.min(gt_cosines),
                        'max_ground_truth_cosine_similarity': np.max(gt_cosines),
                        'sample_count': len(pred_losses)
                    })
        
        # 总体汇总
        for step in self.validation_steps:
            pred_losses = results['step_wise_losses'][step]
            pred_cosines = results['step_wise_cosine_sims'][step]
            gt_losses = results['step_wise_ground_truth_losses'][step]
            gt_cosines = results['step_wise_ground_truth_cosine_sims'][step]
            
            if pred_losses:
                summary_data.append({
                    'env_name': 'ALL_ENVIRONMENTS',
                    'validation_step': step,
                    # 预测损失统计
                    'mean_prediction_loss': np.mean(pred_losses),
                    'std_prediction_loss': np.std(pred_losses),
                    'min_prediction_loss': np.min(pred_losses),
                    'max_prediction_loss': np.max(pred_losses),
                    'mean_prediction_cosine_similarity': np.mean(pred_cosines),
                    'std_prediction_cosine_similarity': np.std(pred_cosines),
                    'min_prediction_cosine_similarity': np.min(pred_cosines),
                    'max_prediction_cosine_similarity': np.max(pred_cosines),
                    # 真值损失统计
                    'mean_ground_truth_loss': np.mean(gt_losses),
                    'std_ground_truth_loss': np.std(gt_losses),
                    'min_ground_truth_loss': np.min(gt_losses),
                    'max_ground_truth_loss': np.max(gt_losses),
                    'mean_ground_truth_cosine_similarity': np.mean(gt_cosines),
                    'std_ground_truth_cosine_similarity': np.std(gt_cosines),
                    'min_ground_truth_cosine_similarity': np.min(gt_cosines),
                    'max_ground_truth_cosine_similarity': np.max(gt_cosines),
                    'sample_count': len(pred_losses)
                })
        
        # 保存汇总CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, f"summary_results_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"汇总CSV已保存至: {summary_file}")
        
        return summary_df
    
    def generate_report(self, results: Dict, output_dir: str, timestamp: str):
        """生成验证报告"""
        report_file = os.path.join(output_dir, f"validation_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("PRM_JE 模型推理验证报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"验证步数: {self.validation_steps}\n")
            f.write(f"每环境回合数: {self.num_episodes}\n")
            f.write(f"环境名称: {results.get('env_names', ['未知'])}\n\n")
            
            f.write("损失计算说明:\n")
            f.write("-" * 30 + "\n")
            f.write("1. 预测损失: 模型预测特征 vs 真实环境特征\n")
            f.write("2. 真值损失: 初始图像特征 vs 各步真实环境特征\n\n")
            
            # 总体统计
            f.write("总体统计:\n")
            f.write("-" * 30 + "\n")
            for step in self.validation_steps:
                pred_losses = results['step_wise_losses'][step]
                pred_cosines = results['step_wise_cosine_sims'][step]
                gt_losses = results['step_wise_ground_truth_losses'][step]
                gt_cosines = results['step_wise_ground_truth_cosine_sims'][step]
                
                if pred_losses:
                    f.write(f"步数 {step:2d}:\n")
                    f.write(f"  预测损失: {np.mean(pred_losses):.6f} ± {np.std(pred_losses):.6f}\n")
                    f.write(f"  预测余弦相似度: {np.mean(pred_cosines):.4f} ± {np.std(pred_cosines):.4f}\n")
                    f.write(f"  真值损失: {np.mean(gt_losses):.6f} ± {np.std(gt_losses):.6f}\n")
                    f.write(f"  真值余弦相似度: {np.mean(gt_cosines):.4f} ± {np.std(gt_cosines):.4f}\n")
                    f.write(f"  样本数量: {len(pred_losses)}\n\n")
            
            # 按环境统计
            f.write("按环境统计:\n")
            f.write("-" * 30 + "\n")
            for env_idx, env_result in enumerate(results['env_results']):
                env_name = env_result.get('env_name', f'Env_{env_idx}')
                f.write(f"环境 {env_idx + 1} ({env_name}):\n")
                for step in self.validation_steps:
                    pred_losses = env_result['losses'][step]
                    pred_cosines = env_result['cosine_similarities'][step]
                    gt_losses = env_result['ground_truth_losses'][step]
                    gt_cosines = env_result['ground_truth_cosine_similarities'][step]
                    
                    if pred_losses:
                        f.write(f"  步数 {step:2d}: \n")
                        f.write(f"    预测 - Loss={np.mean(pred_losses):.6f}, Cosine={np.mean(pred_cosines):.4f}\n")
                        f.write(f"    真值 - Loss={np.mean(gt_losses):.6f}, Cosine={np.mean(gt_cosines):.4f}\n")
                f.write("\n")
        
        print(f"报告已保存至: {report_file}")
    
    def plot_results(self, results: Dict, output_dir: str, timestamp: str):
        """生成可视化图表"""
        plt.figure(figsize=(20, 15))
        
        # 预测损失随步数变化
        plt.subplot(3, 2, 1)
        step_means = []
        step_stds = []
        for step in self.validation_steps:
            losses = results['step_wise_losses'][step]
            if losses:
                step_means.append(np.mean(losses))
                step_stds.append(np.std(losses))
            else:
                step_means.append(0)
                step_stds.append(0)
        
        plt.errorbar(self.validation_steps, step_means, yerr=step_stds, 
                    marker='o', capsize=5, capthick=2, label='预测损失')
        plt.xlabel('预测步数')
        plt.ylabel('平均预测损失')
        plt.title('预测损失随预测步数变化')
        plt.grid(True)
        plt.legend()
        
        # 预测余弦相似度随步数变化
        plt.subplot(3, 2, 2)
        cosine_means = []
        cosine_stds = []
        for step in self.validation_steps:
            cosines = results['step_wise_cosine_sims'][step]
            if cosines:
                cosine_means.append(np.mean(cosines))
                cosine_stds.append(np.std(cosines))
            else:
                cosine_means.append(0)
                cosine_stds.append(0)
        
        plt.errorbar(self.validation_steps, cosine_means, yerr=cosine_stds, 
                    marker='s', capsize=5, capthick=2, color='green', label='预测余弦相似度')
        plt.xlabel('预测步数')
        plt.ylabel('预测余弦相似度')
        plt.title('预测余弦相似度随预测步数变化')
        plt.grid(True)
        plt.legend()
        
        # 真值损失随步数变化
        plt.subplot(3, 2, 3)
        gt_step_means = []
        gt_step_stds = []
        for step in self.validation_steps:
            gt_losses = results['step_wise_ground_truth_losses'][step]
            if gt_losses:
                gt_step_means.append(np.mean(gt_losses))
                gt_step_stds.append(np.std(gt_losses))
            else:
                gt_step_means.append(0)
                gt_step_stds.append(0)
        
        plt.errorbar(self.validation_steps, gt_step_means, yerr=gt_step_stds, 
                    marker='o', capsize=5, capthick=2, color='red', label='真值损失')
        plt.xlabel('预测步数')
        plt.ylabel('平均真值损失')
        plt.title('真值损失随预测步数变化')
        plt.grid(True)
        plt.legend()
        
        # 真值余弦相似度随步数变化
        plt.subplot(3, 2, 4)
        gt_cosine_means = []
        gt_cosine_stds = []
        for step in self.validation_steps:
            gt_cosines = results['step_wise_ground_truth_cosine_sims'][step]
            if gt_cosines:
                gt_cosine_means.append(np.mean(gt_cosines))
                gt_cosine_stds.append(np.std(gt_cosines))
            else:
                gt_cosine_means.append(0)
                gt_cosine_stds.append(0)
        
        plt.errorbar(self.validation_steps, gt_cosine_means, yerr=gt_cosine_stds, 
                    marker='s', capsize=5, capthick=2, color='orange', label='真值余弦相似度')
        plt.xlabel('预测步数')
        plt.ylabel('真值余弦相似度')
        plt.title('真值余弦相似度随预测步数变化')
        plt.grid(True)
        plt.legend()
        
        # 预测损失vs真值损失对比
        plt.subplot(3, 2, 5)
        plt.plot(self.validation_steps, step_means, 'o-', label='预测损失', linewidth=2)
        plt.plot(self.validation_steps, gt_step_means, 's-', label='真值损失', linewidth=2, color='red')
        plt.xlabel('预测步数')
        plt.ylabel('平均损失')
        plt.title('预测损失 vs 真值损失对比')
        plt.grid(True)
        plt.legend()
        
        # 预测余弦相似度vs真值余弦相似度对比
        plt.subplot(3, 2, 6)
        plt.plot(self.validation_steps, cosine_means, 's-', label='预测余弦相似度', linewidth=2, color='green')
        plt.plot(self.validation_steps, gt_cosine_means, 'o-', label='真值余弦相似度', linewidth=2, color='orange')
        plt.xlabel('预测步数')
        plt.ylabel('余弦相似度')
        plt.title('预测余弦相似度 vs 真值余弦相似度对比')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(output_dir, f"validation_plots_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存至: {plot_file}")


def main():
    """主函数"""
    # 配置
    MODEL_PATH = "Output/episode5/train/model_epoch_6.pth"  # 替换为你的模型路径
    OUTPUT_DIR = "Output/episode5/eval/MultiStepEval_6"  # 输出目录
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请将 MODEL_PATH 替换为你的实际模型路径")
        return
    
    # 创建推理器
    inferencer = PRM_Inference(MODEL_PATH)
    
    print("开始模型推理验证...")
    print(f"验证配置:")
    print(f"  - 验证步数: {inferencer.validation_steps}")
    print(f"  - 最大推理步数: {inferencer.max_steps}")
    print(f"  - 每环境回合数: {inferencer.num_episodes}")
    
    # 运行验证 - 现在固定使用4种特定游戏
    try:
        results = inferencer.run_validation()  # 不需要传参数，内部固定使用4种游戏
        
        # 保存结果
        inferencer.save_results(results, OUTPUT_DIR)
        
        # 打印最终统计
        print("\n" + "="*50)
        print("验证完成! 最终统计:")
        print("="*50)
        
        for step in inferencer.validation_steps:
            pred_losses = results['step_wise_losses'][step]
            pred_cosines = results['step_wise_cosine_sims'][step]
            gt_losses = results['step_wise_ground_truth_losses'][step]
            gt_cosines = results['step_wise_ground_truth_cosine_sims'][step]
            
            if pred_losses:
                print(f"步数 {step:2d}:")
                print(f"  预测损失={np.mean(pred_losses):.6f}, "
                      f"预测余弦相似度={np.mean(pred_cosines):.4f}")
                print(f"  真值损失={np.mean(gt_losses):.6f}, "
                      f"真值余弦相似度={np.mean(gt_cosines):.4f}")
                print(f"  样本数={len(pred_losses)}")
                print()
        
    except KeyboardInterrupt:
        print("\n验证被用户中断")
    except Exception as e:
        print(f"\n验证过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()