import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from env import GridWorldEnv
from agent import MARLAgent, ObservationModule
from target import Target
import random

max_eposides = 1
max_step_count = 1000

def main():
    plt.ion()
    env = GridWorldEnv(length = 3.0, 
                       width = 2.0, 
                       grid_size = 0.05, 
                       num_agent = 32,
                       num_target = 1,
                       num_obstacles = 50)
    #重置环境
    state = env.reset()
    #创建Target
    targets = []
    for i, target_pos in enumerate(env.target_pos):  # 使用env中的目标位置
        target = Target(target_id=i, light_radius=8, max_intensity=1.0)
        target.position = target_pos  # 设置位置
        target.initialize_light_map(env)
        targets.append(target)
    #创建Agent
    agents = []
    for i in range(env.num_agent):
        agent = MARLAgent(agent_id = i, obs_dim = 10)
        agents.append(agent)
    for episode in range(max_eposides):
        state = env.reset()
        done = False
        step_count = 0
        print(f"\nEpisode {episode + 1}")
        # while not done and step_count < max_step_count:
        while step_count < max_step_count:
            actions = []
            for i, agent in enumerate(agents):
                movement_direction = env.agent_directions[i]
                action = agent.choose_action(env, movement_direction)
                actions.append(action)   
            next_state, rewards, done, info = env.step(actions)# 执行动作
            #渲染
            env.render(episode, step_count, agents, targets)
            #存储经验（简化
            for i, agent in enumerate(agents):
                agent.remember(state, actions[i], rewards[i], next_state, done)
            state = next_state
            step_count += 1
            if step_count % 10 == 0:
                total_reward = sum(rewards)
                print(f'Step count: {step_count}, Total Reward: {total_reward}')
            if done:
                print(f"找到目标！在步骤 {step_count}")
                # break
            # 每个episode后学习
            for agent in agents:
                agent.learn()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()