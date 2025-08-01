# env_manager.py
import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py) # 注册 Atari 环境

class AtariEnvManager:
    def __init__(self, num_games, render_mode=None, num_envs=1):
        """
        管理一个或多个 Atari 游戏环境。
        参数:
            num_games (int): 要从预定义游戏中随机选择的游戏数量。
            render_mode (str, optional): 渲染模式 ('human' 或 None)。默认为 None。
            num_envs (int): 要创建的并行环境数量。
        """
        # 预定义的游戏ID列表
        self.available_games = [
            "Seaquest-v4",
            "Riverraid-v4",
            "ChopperCommand-v4"
        ]
        
        # 确保请求的游戏数量不超过可用的游戏数量
        num_games = min(num_games, len(self.available_games))
        
        # 随机选择指定数量的游戏作为游戏池
        self.selected_games = np.random.choice(self.available_games, num_games, replace=False)
        print(f"游戏池中的游戏: {self.selected_games}")
        
        # 存储渲染模式配置
        self.render_mode = render_mode
        # 如果是 'human' 模式，只渲染第一个环境以避免窗口过多
        self.effective_render_mode = [render_mode] + [None] * (num_envs - 1) if render_mode == 'human' else [None] * num_envs
        
        # 为每个环境随机分配一个选中的游戏
        env_games = np.random.choice(self.selected_games, num_envs)
        print(f"尝试创建 {num_envs} 个环境实例")
        for i, game in enumerate(env_games):
            print(f"环境 {i+1} 初始游戏: {game}")

        self.envs = [] # 环境实例列表
        self.current_games = [] # 记录每个环境当前的游戏类型

        for i in range(num_envs):
            try:
                env = gym.make(env_games[i], render_mode=self.effective_render_mode[i])
                self.envs.append(env)
                self.current_games.append(env_games[i])
                print(f"环境实例 {i+1} '{env_games[i]}' 创建成功。动作空间: {env.action_space}")
            except Exception as e:
                print(f"创建环境实例 {i+1} '{env_games[i]}' 失败: {e}")
        
        if not self.envs:
            raise RuntimeError("未能成功创建任何环境实例。")

        self.num_envs = len(self.envs)
        self.current_obs = [None] * self.num_envs # 存储每个环境的当前观测
        self.episode_rewards = [0] * self.num_envs # 存储每个环境当前回合的奖励
        
        # 初始化所有环境的观测值
        for i in range(self.num_envs):
            obs, _ = self.envs[i].reset()
            self.current_obs[i] = obs

    def _create_new_env(self, env_index):
        """
        为指定索引创建新的随机游戏环境
        参数:
            env_index (int): 要重新创建的环境索引
        """
        # 随机选择一个新游戏
        new_game = np.random.choice(self.selected_games)
        
        # 关闭旧环境
        if self.envs[env_index] is not None:
            self.envs[env_index].close()
        
        # 创建新环境
        try:
            new_env = gym.make(new_game, render_mode=self.effective_render_mode[env_index])
            self.envs[env_index] = new_env
            self.current_games[env_index] = new_game
            print(f"环境 {env_index+1} 切换到新游戏: {new_game}")
            return True
        except Exception as e:
            print(f"创建新环境失败 (环境 {env_index+1}, 游戏 {new_game}): {e}")
            return False

    def sample_actions(self):
        """为每个环境采样一个随机动作。"""
        return [env.action_space.sample() for env in self.envs]

    def step(self, actions):
        """
        在每个环境中使用提供的动作执行一步。
        参数:
            actions (list): 动作列表，每个环境一个动作。

        返回:
            experiences (list): (obs1, action, obs2, reward, done) 元组的列表。
                                obs1 和 obs2 是 NumPy 数组。
        """
        experiences = [] # 存储经验元组

        for i in range(self.num_envs):
            env = self.envs[i]
            action = actions[i] # 获取对应环境的动作
            
            obs1 = self.current_obs[i] # 当前观测作为 obs1
            
            # 执行动作，获取新状态和奖励
            obs2, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated # 判断回合是否结束

            experiences.append((obs1, action, obs2, reward, done)) # 存储经验

            self.current_obs[i] = obs2 # 更新当前观测
            self.episode_rewards[i] += reward #累积奖励
            
            if done: # 如果回合结束
                print(f"环境 {i+1}: 回合结束。游戏: {self.current_games[i]}, 总奖励: {self.episode_rewards[i]}")
                
                # 尝试创建新的随机游戏环境
                if self._create_new_env(i):
                    # 重置新环境并获取初始观测
                    self.current_obs[i], _ = self.envs[i].reset()
                    self.episode_rewards[i] = 0 # 重置回合奖励
                else:
                    # 如果创建新环境失败，则重置当前环境
                    print(f"环境 {i+1}: 创建新游戏失败，重置当前游戏 {self.current_games[i]}")
                    self.current_obs[i], _ = env.reset()
                    self.episode_rewards[i] = 0

        return experiences

    def close(self):
        """关闭所有环境。"""
        for env in self.envs:
            if env is not None:
                env.close()
        print(f"{self.num_envs} 个环境已关闭。")

    def get_current_games(self):
        """返回当前每个环境正在运行的游戏"""
        return self.current_games.copy()

if __name__ == '__main__':
    # 示例用法:
    num_parallel_envs = 5  # 并行环境数量
    num_games = 3  # 要随机选择的游戏数量
    num_steps = 1000  # 模拟步数
    
    # 创建环境管理器，随机选择3个游戏
    env_manager = AtariEnvManager(num_games=num_games, num_envs=num_parallel_envs, render_mode='human')

    try:
        for step_num in range(num_steps):
            # 为所有环境采样随机动作
            actions_to_take = env_manager.sample_actions()
            
            # 执行一步并获取经验
            step_experiences = env_manager.step(actions_to_take)
            
            # 每10步打印一次状态
            if step_num % 50 == 0:
                print(f"\nStep {step_num}:")
                current_games = env_manager.get_current_games()
                for i, (obs1, action, obs2, reward, done) in enumerate(step_experiences):
                    print(f"  Env {i+1} ({current_games[i]}): action={action}, reward={reward:.2f}, done={done}")
    
    except KeyboardInterrupt:
        print("\n手动中断模拟")
    
    finally:
        env_manager.close()