import numpy as np
from numpy import *
import torch
import math,time
import random

class rule_control(object):
    n_w = 21   #21
    n_h = n_w
    d_max = n_w
    fd_max = 10

    len_in_window = 10
    
    def __init__(self):
        self.len_in_window = self.len_in_window
        
        self.ele_all = []
        for i in range(1,self.n_w-1):
             for j in range(1,self.n_w-1):
                 self.ele_all.append([i,j])

    
    def bd_com(self,pos):
        bd1 = pos[0] - 0
        bd2 = self.n_w - 1 - pos[0]
        bd3 = pos[1] - 0
        bd4 = self.n_h - 1 - pos[1]
        bd_all = np.array([bd1,bd2,bd3,bd4])
        bd_1 = [bd1,bd2,bd3,bd4]
        bd_min = min(bd_1)
        if bd_min <= 1:
            bd_min_index = bd_1.index(bd_min)
        else:
            bd_min = self.d_max/2
            bd_min_index = 4

        bd_input = [bd_min/self.d_max,bd_min_index/4]
        return bd_all, bd_input
    
    def bd_input_com(self,pos):
        bd1 = pos[0] - 0
        bd2 = self.n_w - 1 - pos[0]
        bd3 = pos[1] - 0
        bd4 = self.n_h - 1 - pos[1]
        bd_1 = [bd1,bd2,bd3,bd4]
        bd_2 = sorted([(value, index) for index, value in enumerate(bd_1)])

        bd_min1, bd_index1 = bd_2[0]
        bd_min2, bd_index2 = bd_2[1]
        
        if bd_min1 >= 2:
            bd_min1 = self.d_max / 2
            bd_index1 = 4
        if bd_min2 >= 2:
            bd_min2 = self.d_max / 2
            bd_index2 = 4
        
        bd_input = [bd_min1/self.d_max, bd_min2/self.d_max, bd_index1/4, bd_index2/4]

        return bd_input
    
    def distance(self,pos1,pos2):
        d_x = pos1[0] - pos2[0]
        d_y = pos1[1] - pos2[1]
        d_min = min(abs(d_x),abs(d_y))
        d_max = max(abs(d_x),abs(d_y))
        d1 = math.sqrt(d_min**2+d_min**2)
        d2 = d_max - d_min
        d_all = d1 + d2
        alpha = math.atan2(d_y,d_x)
        return d_all,alpha

    def step_distance(self,pos1,pos2):
        d_x = pos1[0] - pos2[0]
        d_y = pos1[1] - pos2[1]
        d_step = max(abs(d_x),abs(d_y))
        return d_step
    
    def angle(self, v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180/math.pi)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180/math.pi)
        if angle1*angle2 >= 0:
            included_angle = abs(angle1-angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle

    def state_com(self, pos, fd, in_conti_window):
        input_conti = []
        n_list = len(in_conti_window)
        for i in range(self.len_in_window):
            fd_i = in_conti_window[i%n_list][0]/self.fd_max
            theta_i = in_conti_window[i%n_list][1]/(2*math.pi)
            input_conti.extend([fd_i,theta_i])
        bd_input_1 = self.bd_input_com(pos)
        
        state = bd_input_1 + [fd/self.fd_max] + input_conti
        return state


    def reward_com(self,state_next,done):
        r_cp = -0.02
        #r_cp = 0
        r_end = 0
        r_goal = 0
        r_bound = 0

        fd_pre = (state_next[-4])*self.fd_max
        fd_now = (state_next[-2])*self.fd_max
        fd_change = fd_pre - fd_now
        
        if done:
            r_end = 20

        if not done:
            if fd_change > 0:
                r_goal = fd_change * 0.3
            else:
                r_goal = fd_change * 0.3
            bd_all = state_next[0]
            if bd_all == 0:
                r_bound = -0.5
        reward_all = r_end + r_goal + r_cp + r_bound
        return reward_all

    def reward_sparse(self,state_next,done):
        r_cp = -0.02
        r_end = 0
        r_bound = 0
        if done:
            r_end = 20
        if not done:
            bd_all = state_next[0]
            if bd_all == 0:
                r_bound = -0.5

        reward_all = r_end + r_bound + r_cp
        return reward_all


    def global_state(self, pos_new_whole):
        num_map = np.zeros([self.n_w,self.n_w])
        n_min = int(self.n_w/3)-1
        n_max = 2*int(self.n_w/3)-1
        area_num = [0]*9
        for i in range(len(pos_new_whole)):
            m,n = pos_new_whole[i]
            num_map[m,n] = 1
            if n <= n_min :
                if m <= n_min:
                    area_num[0] += 1
                    
                if m > n_min and m <= n_max:
                    area_num[1] += 1

                if m > n_max:
                    area_num[2] += 1

            if n > n_min and n <= n_max:
                if m <= n_min:
                    area_num[3] += 1

                if m > n_min and m <= n_max:
                    area_num[4] += 1

                if m > n_max:
                    area_num[5] += 1

            if n > n_max:
                if m <= n_min:
                    area_num[6] += 1

                if m > n_min and m <= n_max:
                    area_num[7] += 1

                if m > n_max:
                    area_num[8] += 1

        pos_noreach = self.ele_all[:]
        for i in self.ele_all:
            if i in pos_new_whole:
                pos_noreach.remove(i)

        num_index = []
        min_num = min(area_num)
        for i in range(len(area_num)):
            if area_num[i] == min_num:
                num_index.append(i)
        return num_index, pos_noreach
    


    def center_com(self, pos_now, area_index, pos_noreach):
        n_min = int(self.n_w/3)-1
        n_max = 2*int(self.n_w/3)-1
        pos_center_list = [[3,3],[10,3],[17,3],[3,10],[10,10],[17,10],[3,17],[10,17],[17,17]]

        index = np.random.choice(len(area_index))
        mind_index = area_index[index]
        random.shuffle(pos_noreach)
        
        for i in range(len(pos_noreach)):
            pos = pos_noreach[i]
            pos_center1 = []
            bd_all ,_ = self.bd_com(pos)
            if min(bd_all) >= 0:
                if mind_index == 0:
                    if pos[1] <= n_min and pos[0] <= n_min:
                        pos_center1 = pos
                        break
                if mind_index == 1:
                    if pos[1] <= n_min and pos[0] > n_min and pos[0] <= n_max:
                        pos_center1 = pos
                        break
                if mind_index == 2:
                    if pos[1] <= n_min and pos[0] > n_max:
                        pos_center1 = pos
                        break
                if mind_index == 3:
                    if pos[1] > n_min and pos[1] <= n_max and pos[0] <= n_min:
                        pos_center1 = pos
                        break
                if mind_index == 4:
                    if pos[1] > n_min and pos[1] <= n_max and pos[0] > n_min and pos[0] <= n_max:
                        pos_center1 = pos
                        break
                if mind_index == 5:
                    if pos[1] > n_min and pos[1] <= n_max and pos[0] > n_max:
                        pos_center1 = pos
                        break
                if mind_index == 6:
                    if pos[1] > n_max and pos[0] <= n_min:
                        pos_center1 = pos
                        break
                if mind_index == 7:
                    if pos[1] > n_max and pos[0] > n_min and pos[0] <= n_max:
                        pos_center1 = pos
                        break
                if mind_index == 8:
                    if pos[1] > n_max and pos[0] > n_max:
                        pos_center1 = pos
                        break
            if i == len(pos_noreach)-1 and pos_center1 == []:
                pos_center1 = pos_center_list[mind_index]
                pos_center1[0] += (random.randint(0,6) - 3)
                pos_center1[1] += (random.randint(0,6) - 3)
        return pos_center1


    def action_com(self, area_index, pos_noreach, pos_now, pos_center):
         if pos_center == []:
             pos_center = self.center_com(pos_now, area_index, pos_noreach)

         posy = pos_center[1] - pos_now[1]
         posx = pos_center[0] - pos_now[0]

         if posy != 0 and posx != 0:
             if posy > 0 and posx > 0:
                 action = 3
             if posy > 0 and posx < 0:
                 action = 5
             if posy < 0 and posx < 0:
                 action = 7
             if posy < 0 and posx > 0:
                 action = 1
             num_conti = min(abs(posy),abs(posx))
         if posy == 0:
             if posx > 0:
                 action = 2
             if posx < 0:
                 action = 6
             num_conti = abs(posx)
         if posx == 0:
             if posy > 0:
                 action = 4
             if posy < 0:
                 action = 0
             num_conti = abs(posy)
         if posy == 0 and posx == 0:
             action = np.random.randint(0, 8)
             num_conti = 4

         return action, num_conti, pos_center
