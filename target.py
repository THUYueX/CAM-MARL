import numpy as np
class Target:
    def __init__(self, target_id, light_radius=5, max_intensity=1):
        self.target_id = target_id
        self.light_radius = light_radius
        self.max_intensity = max_intensity
        self.position = None
    def initialize_light_map(self, env):
        """初始化光照地图"""
        if self.position is None:
            return
        x, y = self.position
        for dx in range(-self.light_radius, self.light_radius + 1):
            for dy in range(-self.light_radius, self.light_radius + 1):
                nx, ny = x + dx, y + dy
                # 边界检查
                if not (0 <= nx < env.n_length and 0 <= ny < env.n_width):
                    continue
                distance = np.sqrt(dx ** 2 + dy ** 2)
                if distance > self.light_radius:
                    continue
                intensity = self.max_intensity * (1 - distance / self.light_radius)
                env.light_score_map[nx, ny] += intensity