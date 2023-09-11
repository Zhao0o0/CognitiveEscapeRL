import gym
from gym import register
register(
    id='MyMultiRoomEnv-v0',
    entry_point='new_multiroom_env:MyMultiRoomEnv'
)

import matplotlib.pyplot as plt
from gym_minigrid.wrappers import RGBImgObsWrapper
from noisy_tv_wrapper import NoisyTVWrapper
from collections import deque
import random


class Position_memory:
    def __init__(self):
        self.current_position = (0, 0)
        self.current_direction = 'up' # 假设智能体初始面向上方
        self.visited_positions = {self.current_position: True}
        #self.num_visited_positions_history = deque(maxlen=10)

    def move(self, action):
        if action == 0:     ###'左转':
            self.turn_left()
        elif action == 1:      ###'右转':
            self.turn_right()
        elif action == 2:     ###'前进':
            self.move_forward()
            self.visited_positions[self.current_position] = True
        print('current_position:{}',format(self.current_position))
        
        num_visited_positions = len(self.visited_positions)
        return num_visited_positions


    def turn_left(self):
        if self.current_direction == 'up':
            self.current_direction = 'left'
        elif self.current_direction == 'down':
            self.current_direction = 'right'
        elif self.current_direction == 'left':
            self.current_direction = 'down'
        elif self.current_direction == 'right':
            self.current_direction = 'up'

    def turn_right(self):
        if self.current_direction == 'up':
            self.current_direction = 'right'
        elif self.current_direction == 'down':
            self.current_direction = 'left'
        elif self.current_direction == 'left':
            self.current_direction = 'up'
        elif self.current_direction == 'right':
            self.current_direction = 'down'

    def move_forward(self):
        x, y = self.current_position
        if self.current_direction == 'up':
            self.current_position = (x, y + 1)
        elif self.current_direction == 'down':
            self.current_position = (x, y - 1)
        elif self.current_direction == 'left':
            self.current_position = (x - 1, y)
        elif self.current_direction == 'right':
            self.current_position = (x + 1, y)


env = gym.make('MyMultiRoomEnv-v0')
position_memory = Position_memory()
env = NoisyTVWrapper(env, "True")
obs_space = env.observation_space
print(obs_space)

observation, info = env.reset(seed=42)
action_list = [0,2,1,2,1,2,1,2,1,2,1]

max_deque = 6
max_escape_step = 8
pos_now = env.agent_pos
num_explore_list = []
num_visited_positions_history = deque(maxlen=max_deque)
local_optimum = False
num_escape_step = 1

for _ in range(10000):
    '''
    while True:
        action = env.action_space.sample()  # User-defined policy function
        if action in action_list:
            break
    '''
    i = 0
    while True:
        #正常导航
        if not local_optimum:
            #action = action_list[i]
            action = env.action_space.sample()
        #执行逃逸策略
        else:
            action = random.choice([0,1,2])
            num_escape_step += 1
            print('num_escape_step',num_escape_step)
            if num_escape_step >= max_escape_step:
                local_optimum = False
                num_escape_step = 1
                num_visited_positions_history = deque(maxlen=max_deque)
                
        observation, reward, terminated, truncated, info = env.step(action)
        i += 1
        if i == len(action_list):
            i = 0
    #plt.imshow(observation['image'])
    #env.render("human")
        #print(env.agent_pos)
        print('action',action)
        
        pos_next = env.agent_pos
        if action==2 and pos_now == pos_next:
            print('barrier!!!')
        else:
            num_visited_positions = position_memory.move(action)
            
        num_visited_positions_history.append(num_visited_positions)
        
        print('num_visited_positions',num_visited_positions)
        print("len(num_visited_positions_history)",len(num_visited_positions_history))
        
        if len(num_visited_positions_history) >= max_deque:
            first = num_visited_positions_history[0]
            last = num_visited_positions_history[-1]
            num_difference = last - first
        else:
            num_difference = 10
        
        print("num_difference",num_difference)
        
        ##判断是否一直处于一个局部最优
        if num_difference <= 3:
            local_optimum = True
            #将动作限制到[0,1,2]之间
            
        if terminated or truncated:
            observation, info = env.reset()
            position_memory = Position_memory()
            i = 0
            num_explore_list = []
            num_visited_positions_history = deque(maxlen=max_deque)
            local_optimum = False
            num_escape_step = 1
env.close()










