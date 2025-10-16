import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from sensor import SensorSuite
from collections import deque

class ActorNetwork(nn.Module):
    """演员 - 策略网络(action)"""
    def __init__(self, input_dim, hidden_dims = [128, 64], output_dim = 8):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        x = self.network(x)
        return F.softmax(x, dim = -1) #输出动作概率
    
class CriticNetwork(nn.Module):
    """评论家 - 价值函数网络Q(critic)"""
    def __init__(self, state_dim, action_dim, hidden_dims = [128, 64], output_dim = 1):
        super().__init__()
        input_dim = state_dim + action_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)
    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)
        return self.network(x)

class ValueMapNetwork(nn.Module):
    """价值地图生成网络dnn 用于生成地图内在奖励"""
    def __init__(self, input_dim, hidden_dims = [128, 64], output_dim = 1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
    
class MARLAgent:
    """多智能体Agent"""
    def __init__(self, agent_id, obs_dim, lr = 0.001, gamma = 0.5):
        self.agent_id = agent_id

        self.value_net = ValueMapNetwork(obs_dim)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr = lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.obs_module = ObservationModule(agent_id)
        self.comm_module = CommunicationModule(gamma)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        #经验缓存
        self.memory = deque(maxlen = 5000)
        self.batch_size = 32
    def choose_action(self, env, movement_direction):
        observation = self.obs_module.get_observations(env, movement_direction)
        #使用epsilon贪心策略
        if random.random() < self.epsilon:
            return random.randint(0, 7)
        else:
            return random.randint(0, 7)
    
    def _network_based_action(self, env, observation):
        #将观测转换为网络输入
        obs_vector = self._observation_to_vector(observation)

    def _observation_to_vector(self, observation, env):
        return torch.tensor(observation).float()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        #TODO 实现学习逻辑

        #衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class ObservationModule:
    """观测-感知模块  不属于框架内容"""
    def __init__(self, agent_id, obstacle_perception_radius = 10, light_perception_radius = 1,  ir_perception_radius = 2):
        """初始化智能体"""
        self.agent_id = agent_id
        # 初始化传感器套件
        self.sensor_suite = SensorSuite(obstacle_perception_radius, light_perception_radius, ir_perception_radius)
        # 设置各感知半径
        self.obstacle_perception_radius = obstacle_perception_radius
        self.light_perception_radius = light_perception_radius
        self.ir_perception_radius = ir_perception_radius

        # 方向感知映射表，键为方向编号，值为该方向可感知的坐标偏移列表
        self.direction_perception = {
            0: [(0, 1), (-1, -1), (-1, 1)],    # 向上
            1: [(1, 1), (-1, 0), (0, -1)],     # 右上
            2: [(1, 0), (-1, 1), (-1, -1)],    # 向右
            3: [(1, -1), (0, 1), (-1, 0)],     # 右下
            4: [(0, -1), (1, 1), (-1, 1)],     # 向下
            5: [(-1, -1), (1, 0), (0, 1)],     # 左下
            6: [(-1, 0), (1, -1), (1, 1)],     # 向左
            7: [(-1, 1), (0, -1), (1, 0)],     # 左上
        }

    def get_observations(self, env, movement_direction):
        self_pos = env.agent_pos[self.agent_id]
        local_neighbor_info = self._get_neighbor_info(env, self_pos)
        local_light_info = self._get_local_light_intensity(env, self_pos, movement_direction)
        local_obstacles_info = self._get_local_obstacles(env, self_pos, movement_direction)
        obs = {
            'agent_id': self.agent_id,
            'self_position': self_pos,
            'light_intensity': local_light_info,
            'local_obstacles': local_obstacles_info,
            'neighbor_info': local_neighbor_info
        }
        return obs
    
    #TODO 获取移动方向
    def _get_movement_direction(self, direction):
        # TODO 摄像头拍摄/速度传感器接口
        return random.randint(0, 7)

    def _get_neighbor_info(self, env, self_pos): 
        x, y = self_pos
        neighbors = []
        for i, agents_pos in enumerate(env.agent_pos):
            if i == self.agent_id:
                continue
            ox, oy = agents_pos
            distance = ((ox - x) ** 2 + (oy - y) ** 2) ** 0.5
            if distance <= self.ir_perception_radius:
                neighbors.append((i, distance))
        return neighbors
    def _get_local_obstacles(self, env, self_pos, movement_direction):
        x, y = self_pos
        agent_direction = movement_direction
        three_prection_directions = self.direction_perception.get(agent_direction)
        for dir_dx, dir_dy in three_prection_directions:
            for distance in range(1,  self.obstacle_perception_radius + 1):
                nx, ny = x + dir_dx * distance, y + dir_dy * distance
                if 0 <= nx < env.n_length and 0 <= ny < env.n_width:
                    obstacle_value = self.sensor_suite.laser_sensor.read_obstacle_info(env, x, y, dir_dx, dir_dy, distance)
                    if obstacle_value > env.obstacle_score_map[nx, ny]:
                        env.obstacle_score_map[nx, ny] = obstacle_value

    def _get_local_light_intensity(self, env, self_pos, movement_direction):
        x, y = self_pos
        agent_direction = movement_direction
        three_prection_directions = self.direction_perception.get(agent_direction)
        for dir_dx, dir_dy in three_prection_directions:
            for distance in range(1,  self.obstacle_perception_radius + 1):
                nx, ny = x + dir_dx * distance, y + dir_dy * distance
                # env.light_score_map[nx][ny] = self.sensor_suite.light_sensor.read_light_info(env, x, y, dir_dx, dir_dy, distance)
   
    
    def _get_local_ir_intensity(self, env, self_pos, movement_direction):
        pass

class CommunicationModule:
    """通信模块 - 处理邻居信息融合"""
    def __init__(self, gamma = 0.7):
        self.gamma = gamma
    def fuse_information(self, self_obs, neighbor_obs_list):
        if not neighbor_obs_list:
            return self_obs
        #TODO 目前是加权平均融合，感觉可以用attention机制 这里就是要着重考虑的地方
        neighbor_obs_array = np.array(neighbor_obs_list)
        fused_neighbor_info = np.mean(neighbor_obs_array, axis = 0)
        fused_obs = self_obs + self.gamma * fused_neighbor_info
        return fused_obs