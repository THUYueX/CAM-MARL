import numpy as np

class LaserSensor:
    """激光传感器 - 用于障碍物检测"""
    def __init__(self, perception_radius):
        self.perception_radius = perception_radius
    
    def read_obstacle_info(self, env, x, y, dir_dx, dir_dy, distance):
        """读取指定方向的障碍物信息 - 返回梯度值"""
        check_x, check_y = x + dir_dx * distance, y + dir_dy * distance
        
        # 首先检查目标格子本身是否是障碍物或边界
        if not (0 <= check_x < env.n_length and 0 <= check_y < env.n_width):
            return 1.0  # 边界视为障碍物
        
        if env.grid[check_x, check_y] == 1:
            return 1.0  # 目标格子就是障碍物
        
        # 如果目标格子不是障碍物，检查路径上是否有障碍物
        nearest_obstacle_distance = None
        for d in range(1, self.perception_radius + 1):
            temp_x, temp_y = x + dir_dx * d, y + dir_dy * d
            
            # 边界检查
            if not (0 <= temp_x < env.n_length and 0 <= temp_y < env.n_width):
                nearest_obstacle_distance = d
                break
            
            # 障碍物检查
            if env.grid[temp_x, temp_y] == 1:
                nearest_obstacle_distance = d
                break
        
        # 计算梯度值
        if nearest_obstacle_distance is not None and distance <= nearest_obstacle_distance:
            return distance / nearest_obstacle_distance  # 距离障碍物越近，值越大
        else:
            return 0.0  # 无障碍物
    
    #获取障碍梯度
    def get_obstacle_gradient(self, env, start_x, start_y, dir_dx, dir_dy):
        max_obstacle_range = self.perception_radius
        gradient_profile = [0.0] * max_obstacle_range
        nearest_obstacle_distance = None
        
        # 找到最近的障碍物(包括边界)
        for distance in range(1, max_obstacle_range + 1):
            check_x, check_y = start_x + distance * dir_dx, start_y + distance * dir_dy
            if not (0 <= check_x < env.n_length and 0 <= check_y < env.n_width):
                nearest_obstacle_distance = distance
                break
            elif env.grid[check_x, check_y] == 1:
                nearest_obstacle_distance = distance
                break
        
        # 修复：这个if应该在for循环外面
        if nearest_obstacle_distance is not None:
            for distance in range(1, nearest_obstacle_distance + 1):
                normalized_distance = distance / nearest_obstacle_distance
                gradient_profile[distance - 1] = normalized_distance
        
        return gradient_profile

class IRSensor:
    """红外传感器 - 用于检测其他智能体"""
    def __init__(self, perception_radius):
        self.perception_radius = perception_radius
    
    def read_ir_info(self, env, x, y, agent_id):
        """读取周围红外信息"""
        ir_intensity = 0.0
        
        for dx in range(-self.perception_radius, self.perception_radius + 1):
            for dy in range(-self.perception_radius, self.perception_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                check_x, check_y = x + dx, y + dy
                
                if not (0 <= check_x < env.n_length and 0 <= check_y < env.n_width):
                    continue
                
                # 检测该位置是否有其他智能体
                for i, agent_pos in enumerate(env.agent_pos):
                    if i == agent_id:
                        continue
                    ax, ay = agent_pos
                    if ax == check_x and ay == check_y:
                        # 距离越近，信号越强
                        dist = max(abs(dx), abs(dy))
                        intensity = max(0, 1 - dist / self.perception_radius)
                        ir_intensity = max(ir_intensity, intensity)
                        break
        
        return ir_intensity

class LightSensor:
    """光传感器 - 用于检测目标光照"""
    def __init__(self, perception_radius):
        self.perception_radius = perception_radius
    
    def read_light_info(self, env, x, y):
        """读取光照强度"""
        light_intensity = 0.0
        
        for target_pos in env.target_pos:
            tx, ty = target_pos
            distance = ((tx - x) ** 2 + (ty - y) ** 2) ** 0.5
            
            if distance <= self.perception_radius:
                intensity = max(0, 1 - distance / self.perception_radius)
                light_intensity = max(light_intensity, intensity)
        
        return light_intensity

class SensorSuite:
    """传感器套件 - 统一管理所有传感器"""
    def __init__(self, obstacle_radius, ir_radius, light_radius):
        self.laser_sensor = LaserSensor(obstacle_radius)
        self.ir_sensor = IRSensor(ir_radius)
        self.light_sensor = LightSensor(light_radius)