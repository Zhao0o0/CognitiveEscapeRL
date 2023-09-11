import numpy as np
import time,math
import sys
import pickle


UNIT = 30
radius = 10
MAZE_H = 21
MAZE_W = MAZE_H
start_x,start_y,goal_x,goal_y = 0,0,3,12

class Similarity_World(object):
    def __init__(self):
        super(Similarity_World, self).__init__()

        self.action_space = [np.array([0,-1]),np.array([1,-1]),np.array([1,0]),np.array([1,1]),np.array([0,1]),np.array([-1,1]),np.array([-1,0]),np.array([-1,-1])]
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.pos = []
        self.fd_list = np.zeros((MAZE_H,MAZE_W))


    def similarity_map(self,goal_name):
        load_similarity_path = "similarity_map_2km_train/list{}.pt".format(goal_name)
        with open(load_similarity_path, "rb") as f:
            data_list = pickle.load(f)
        self.fd_list = np.array(data_list[2]).reshape(MAZE_H,MAZE_W)
        
        rand = np.random.uniform()
        z = list(np.arange(MAZE_H))
        z1 = z[::-1]
            
            
        if rand <= 0.25:
            self.fd_list = self.fd_list

        if rand > 0.25 and rand <= 0.5:
            self.fd_list = self.fd_list[:,z1]

        if rand > 0.5 and rand <= 0.75:
            self.fd_list = self.fd_list[z1,:]
        
        if rand > 0.75 and rand <= 1:
            self.fd_list = self.fd_list[z1,:]
            self.fd_list = self.fd_list[:,z1]

        fd_min = np.min(self.fd_list)
        min_index = np.where(self.fd_list == fd_min)
        min_index = [min_index[0].tolist()[0],min_index[1].tolist()[0]]
        return min_index,fd_min

    def reset(self,start_pos,goal_name):
        self.goal_pos,fd_min = self.similarity_map(goal_name)
        goal_x,goal_y = self.goal_pos

        self.pos = start_pos
        start_x,start_y = self.pos
        fd = self.fd_list[start_x][start_y]
        
        theta = math.pi
        
        done = False
        if abs(self.pos[0] - self.goal_pos[0])<=1 and abs(self.pos[1] - self.goal_pos[1])<=1:
            done = True
        return self.pos, fd, theta, self.goal_pos, fd_min, done

    def step(self, action):
        done = False
        base_action = np.array([0, 0])
        if action == 0:
            if self.pos[1] > 0:
                base_action[1] -= UNIT
                self.pos = (np.array(self.pos)+self.action_space[0]).tolist()

        elif action == 1:
            if self.pos[1] > 0 and self.pos[0] < (MAZE_W - 1):
                base_action[1] -= UNIT
                base_action[0] += UNIT
                self.pos = (np.array(self.pos)+self.action_space[1]).tolist()

        elif action == 2:
            if self.pos[0] < (MAZE_W - 1):
                base_action[0] += UNIT
                self.pos = (np.array(self.pos)+self.action_space[2]).tolist()

        elif action == 3:
            if self.pos[1] < (MAZE_W - 1) and self.pos[0] < (MAZE_W - 1):
                base_action[1] += UNIT
                base_action[0] += UNIT
                self.pos = (np.array(self.pos)+self.action_space[3]).tolist()

        elif action == 4:
            if self.pos[1] < (MAZE_W - 1):
                base_action[1] += UNIT
                self.pos = (np.array(self.pos)+self.action_space[4]).tolist()

        elif action == 5:
            if self.pos[1] < (MAZE_W - 1) and self.pos[0] > 0:
                base_action[1] += UNIT
                base_action[0] -= UNIT
                self.pos = (np.array(self.pos)+self.action_space[5]).tolist()

        elif action == 6:
            if self.pos[0] > 0:
                base_action[0] -= UNIT
                self.pos = (np.array(self.pos)+self.action_space[6]).tolist()

        elif action == 7:
            if self.pos[1] > 0 and self.pos[0] > 0:
                base_action[1] -= UNIT
                base_action[0] -= UNIT
                self.pos = (np.array(self.pos)+self.action_space[7]).tolist()

        x,y = self.pos
        fd = self.fd_list[x][y]
        theta = action * math.pi/4

        End = False
        if abs(x - self.goal_pos[0])<= 1 and abs(y - self.goal_pos[1])<= 1 and fd < 0.7:
            End = True
            done = True
            print("End:::End:{},done:{},fd:{}".format(End,done,fd))

        return self.pos, fd, theta, done

    def render(self):
        self.update()