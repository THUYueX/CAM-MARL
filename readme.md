CAM-MARL算法伪代码：
循环每个时间步：
  1. 每个机器人独立：
      - 获取当前局部观测 obs_t
      - Actor(obs_t) → 选择动作 a_t
      - 执行动作 a_t → 移动到新格子
  
  2. 进入新格子后立即获得奖励：
      - 环境反馈外在奖励 r_ext（碰撞惩罚、目标奖励等）
      - 价值网络(局部观测 + 全局加权) → 预测内在奖励 r_int
      - 总奖励 r_total = r_ext + λ × r_int
  
  3. 获得新观测 obs_{t+1}（在新格子的感知）
  
  4. 存储经验：(obs_t, a_t, r_total, obs_{t+1}, done, maps_t, global_info_t, global_state_t, global_state_{t+1})
     -----变量完整说明：
        obs_t, obs_{t+1}：局部观测（Actor/Critic用）
        a_t, r_total, done：动作、奖励、终止标志（Actor/Critic用）
        maps_t：三个语义地图（价值网络用）
        global_info_t：全局加权信息（价值网络用）
        global_state_t, global_state_{t+1}：全局状态（价值网络用）

  5. 更新：
      - Critic网络：学习评估局部状态价值
      - Actor网络：基于Critic的评估改进策略
      - 价值网络：学习更好的奖励预测