import numpy as np
import matplotlib.pyplot as plt
import random

class GridWorldEnv:
    def __init__(self, length = 3.0, width = 2.0, grid_size = 0.05, num_agent = 10, num_target = 1, num_obstacles = 4):
        self.length = length
        self.width = width
        self.grid_size = grid_size
        self.n_length = int(length / grid_size)
        self.n_width = int(width / grid_size)
        # 创建空地图 (0=空地, 1=障碍物)
        self.grid = np.zeros((self.n_length, self.n_width))
        #创建障碍物
        self.num_obstacles = num_obstacles
        self.obstacles = []
        #智能体位置坐标
        self.num_agent = num_agent
        self.agent_pos = []
        #target位置坐标
        self.num_target = num_target
        self.target_pos = []
        self.agent_directions = [random.randint(0, 7) for _ in range(num_agent)]  # 每个机器人的运动方向
        self.obstacle_score_map = np.zeros((self.n_length, self.n_width))  # 障碍物梯度地图
        self.light_score_map = np.zeros((self.n_length, self.n_width))  # 光照地图
        # 渲染相关
        self.fig = None
        self.ax = None

        self.generate_obstacles()
        self.place_agents_and_targets()
    
    def generate_obstacles(self):
        obstacles_placed = 0
        max_attempts = self.num_obstacles * 5
        for _ in range(max_attempts):
            if obstacles_placed >= self.num_obstacles:
                break
            start_x = random.randint(0, self.n_length - 1)
            start_y = random.randint(0, self.n_width - 1)
            if self.grid[start_x, start_y] == 0:
                target_size = random.choices([1, 2, 3, 4, 5, 6, 7, 8], 
                                        weights=[20, 20, 15, 15, 10, 10, 5, 5])[0]
                # 使用BFS-like算法生长障碍物
                placed_cells = self._grow_obstacle(start_x, start_y, target_size)
                obstacles_placed += 1

    def _grow_obstacle(self, start_x, start_y, target_size):
        """从种子位置生长障碍物，返回实际放置的格子数"""
        if self.grid[start_x, start_y] == 1:
            return 0
        placed_cells = 0
        cluster_cells = []
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.grid[start_x, start_y] = 1
        cluster_cells.append((start_x, start_y))
        placed_cells += 1
        while placed_cells < target_size and cluster_cells:
            current_cell = random.choice(cluster_cells)
            cx, cy = current_cell
            random.shuffle(directions)
            expanded = False
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.n_length and 0 <= ny < self.n_width and 
                    self.grid[nx, ny] == 0 and placed_cells < target_size):
                    self.grid[nx, ny] = 1
                    cluster_cells.append((nx, ny))
                    placed_cells += 1
                    expanded = True
                    break
            if not expanded:
                cluster_cells.remove(current_cell)
        return placed_cells

    
    def place_agents_and_targets(self):
        empty_positions = []
        for i in range(self.n_length):
            for j in range(self.n_width):
                if self.grid[i, j] == 0:
                    empty_positions.append((i, j))

        total_entities = self.num_agent + self.num_target
        assert len(empty_positions) >= total_entities, \
        f"Not enough empty positions to place {self.num_agent} agents and {self.num_target} targets! Need {total_entities}, have {len(empty_positions)}"
        selected_positions = random.sample(empty_positions, total_entities)
        self.agent_pos = selected_positions[:self.num_agent]
        self.target_pos = selected_positions[-self.num_target:]

    def reset(self):
        """重置环境：重新生成障碍物和智能体位置"""
        # 重新生成障碍物
        self.grid = np.zeros((self.n_length, self.n_width))
        self.generate_obstacles()
        
        # 重置智能体位置和目标位置
        self.agent_pos = []
        self.target_pos = []
        self.agent_directions = [random.randint(0, 7) for _ in range(self.num_agent)]
        
        #重置地图
        self.obstacle_score_map = np.zeros((self.n_length, self.n_width))
        self.light_score_map = np.zeros((self.n_length, self.n_width)) 
        
        # 重新放置智能体和目标
        self.place_agents_and_targets()
        return self.get_global_state()

    def get_global_state(self):
        """获取全局状态（给外部监控用）"""
        return {
            'agent_positions': self.agent_pos.copy(),
            'target_positions': self.target_pos.copy(),
            'obstacle_map': self.grid.copy(),
            'agent_directions': self.agent_directions.copy()
        }

    def render(self, episode = None, step_count = None, agents = None, targets = None):
        if self.fig is None:
            max_dim = max(self.n_length, self.n_width)
            scale = 15 / max_dim
            fig_width = self.n_length * scale
            fig_height = self.n_width * scale
            self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
        
        self.ax.clear()
        self.ax.set_xlim(0, self.n_length)
        self.ax.set_ylim(0, self.n_width)
        self.ax.set_aspect('equal')
        self.ax.set_adjustable('box')
        self.ax.set_xmargin(0)
        self.ax.set_ymargin(0)

        #显示障碍物梯度地图
        if np.max(self.obstacle_score_map) > 0:
            gradient_display = self.ax.imshow(self.obstacle_score_map.T, 
                                            cmap='Purples', alpha=0.8,
                                            extent=[0, self.n_length, 0, self.n_width],
                                            origin='lower', vmin=0.0, vmax=1.0)
        #显示光照地图
        if np.max(self.light_score_map) > 0:
            light_display = self.ax.imshow(self.light_score_map.T, 
                                        cmap='Greens',  # 绿色光谱
                                        alpha=0.2,      # 透明度
                                        extent=[0, self.n_length, 0, self.n_width],
                                        origin='lower', 
                                        vmin=0.0,
                                        vmax=1.0)
            
        for i in range(self.n_length + 1):
            self.ax.axvline(i, color='gray', linewidth=0.5)
        for j in range(self.n_width + 1):
            self.ax.axhline(j, color='gray', linewidth=0.5)
        obstacle_cells = np.argwhere(self.grid == 1)
        for cell in obstacle_cells:
            x, y = cell
            rect = plt.Rectangle((x, y), 1, 1, facecolor='black', alpha=0.7)
            self.ax.add_patch(rect)    
        direction_offsets = {
            0: (0, 1),
            1: (1, 1),  
            2: (1, 0),
            3: (1, -1),
            4: (0, -1),
            5: (-1, -1),
            6: (-1, 0),
            7: (-1, 1)
        }
        for i, (x, y) in enumerate(self.agent_pos):
            color = plt.cm.Set1(i / len(self.agent_pos))
            self.ax.plot(x + 0.5, y + 0.5, 'o', color=color, markersize=8, 
                        label=f'Agent {i+1}')
            direction = self.agent_directions[i]
            dx, dy = direction_offsets.get(direction, (0, 0))
            start_x, start_y = x + 0.5, y + 0.5
            self.ax.arrow(start_x, start_y, 
                        dx * 0.8, dy * 0.8,
                        head_width=0.15, head_length=0.15, 
                        fc=color, ec=color, alpha=0.8, 
                        length_includes_head=True)
            # 绘制红外感知范围
            if agents and i < len(agents):
                ir_radius = agents[i].obs_module.ir_perception_radius
                # 绘制红外感知范围（完整的圆形区域）
                for dx in range(-ir_radius, ir_radius + 1):
                    for dy in range(-ir_radius, ir_radius + 1):
                        # 检查是否在圆形区域内（包括对角线方向）
                        if dx * dx + dy * dy <= ir_radius * ir_radius:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.n_length and 0 <= ny < self.n_width:
                                rect = plt.Rectangle((nx, ny), 1, 1, 
                                                facecolor=color, alpha=0.2,
                                                edgecolor=color, linewidth=1)
                                self.ax.add_patch(rect)

            
        for i, (x, y) in enumerate(self.target_pos):
            self.ax.plot(x + 0.5, y + 0.5, 'g*', markersize=15, 
                        label=f'Target {i+1}')
            # 绘制光照范围（类似agent的红外感知范围）
            if targets and i < len(targets):
                target_obj = targets[i]
                light_radius = target_obj.light_radius
                # 绘制光照范围（完整的圆形区域）
                for dx in range(-light_radius, light_radius + 1):
                    for dy in range(-light_radius, light_radius + 1):
                        # 检查是否在圆形区域内
                        if dx*dx + dy*dy <= light_radius * light_radius:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.n_length and 0 <= ny < self.n_width:
                                # 使用绿色表示光照范围
                                rect = plt.Rectangle((nx, ny), 1, 1, 
                                                facecolor='green', alpha=0.1,  # 更浅的透明度
                                                edgecolor='green', linewidth=0.5)
                                self.ax.add_patch(rect)

        self.ax.set_xlabel('Length Direction')
        self.ax.set_ylabel('Width Direction')
        self.ax.set_title(f'Grid World - {self.length}m x {self.width}m -> {self.n_length} x {self.n_width} Grid: Episode {episode + 1} - StepCount {step_count + 1}')
        self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.ax.grid(True, alpha=0.3)
        plt.draw()
        plt.pause(0.5)

    def step(self, actions):
        """
        执行所有智能体的动作
        actions: 动作列表，每个元素对应一个智能体的动作 (0-7)
        """
        rewards = [0.0] * self.num_agent
        new_agent_pos = self.agent_pos.copy()
        done = False
        action_to_direction = {
            0: (0, 1),    # 上
            1: (1, 1),    # 右上
            2: (1, 0),    # 右
            3: (1, -1),   # 右下
            4: (0, -1),   # 下
            5: (-1, -1),  # 左下
            6: (-1, 0),   # 左
            7: (-1, 1)    # 左上
        }
        for i, action in enumerate(actions):
            if action < 0 or action > 7:
                continue
            x, y = self.agent_pos[i]
            dx, dy = action_to_direction[action]
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.n_length and 0 <= new_y < self.n_width and 
                self.grid[new_x, new_y] == 0):
                new_agent_pos[i] = (new_x, new_y)
                self.agent_directions[i] = action
                #获得基础移动奖励
                rewards[i] += 0.01
            else:
                # 无效移动(撞墙或者出界)
                rewards[i] -= 0.1

        self.agent_pos = new_agent_pos

        #检查是否到达既定目标
        for i, pos in enumerate(self.agent_pos):
            if pos in self.target_pos:
                rewards[i] += 1.0
                done = True
        info = {
            'collisions': [self.grid[pos[0], pos[1]] == 1 for pos in self.agent_pos],
            'at_target': [pos in self.target_pos for pos in self.agent_pos]
        }
        return self.get_global_state(), rewards, done, info

