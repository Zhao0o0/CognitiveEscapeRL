import numpy as np
from numpy import *
import torch
from d3qn_env import Similarity_World
from d3qn import D3QN_Nstep

from reply_buffer import replay_buffer
from control import rule_control
from torch.utils.tensorboard import SummaryWriter
import math,random
import datetime
import time
import os,sys
import pandas as pd


df_path = "train_result/train_reward{}_{}_{}_{}.csv".format(datetime.datetime.now().month,datetime.datetime.now().day,datetime.datetime.now().hour,datetime.datetime.now().minute)
file_path = "similarity_map_2km_train"

datanames = os.listdir(file_path)
goal_set = []
for i in datanames:
    str1 = i.split('t')[1]
    str2 = str1.split('.')[0]
    goal_set.append(str2)

len_mean_window = 1
len_in_window = 10
len_cycle = 50

n_w = 21
len_in_window = 10
n_state1 = 4 + 1 + 2 * len_in_window
n_action1 = 8
batch_size1 = 512


capacity = 1000000
n_step = 1
gamma_D3QN = 0.99
max_step_all = 5000000
max_episode_step = 380
replay_train_size = 5000
test_step_size = 8000

epsilon_init = 1
epsilon_min = 0.1
final_explore_step = 600000

random_step = 0
levy_flight = False
traversal = False

get_out_local = True
load_model = False
save_model = True
state_dim = (n_state1, )

env = Similarity_World()
rule_control = rule_control()
writer = SummaryWriter('runs/{}_CRL'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

if not os.path.exists('agent1_model/'):
    os.makedirs('agent1_model/')

class Experience:
    def __init__(self, state_now, action, reward, state_next, mask_done):
        self.state_now = state_now
        self.action = action
        self.reward = reward
        self.state_next = state_next
        self.done = mask_done


def run(epoch):
    agent_local = D3QN_Nstep(n_state1,n_action1)
    '''
    if load_model:
        agent_local.save_load_checkpoint("load",load_agent_path)
        print("loading model success!")
    '''
    
    memory1 = replay_buffer(capacity, state_dim)

    num_step_all = 1
    num_conti = 1
    i_episode = 0
    train_step = -1
    epsilon = epsilon_init

    while num_step_all <= max_step_all:
        num_step_episode = 1
        reward_all = 0
        done = False
        get_local = False

        while True:
            goal_name = goal_set[np.random.choice(len(goal_set))]
            start_pos = [random.randrange(4,n_w-5,2), random.randrange(4,n_w-5,2)]
            pos_now, fd_now, theta_now, goal_pos, fd_goal, done = env.reset(start_pos,goal_name)

            pos_new_whole = [pos_now]
            list_len_pos_new = []

            if not done:
                action = 0
                pos_now,fd_now,theta_now,done = env.step(action)
                in_conti_window = [[fd_now,theta_now]]
                pos_new_whole.append(pos_now)
                state_now = rule_control.state_com(pos_now,fd_now,in_conti_window)
            if not done:
                break

        d_s2g,_ = rule_control.distance(start_pos, goal_pos)
        #submax_action = False
        max_local_step = 18
        

        while not done and num_step_episode <= max_episode_step:
            action = agent_local.choose_action(state_now, epsilon)
            num_conti = 1 
            for i in range(num_conti):
                pos_now,fd_now,theta_now,done = env.step(action)
                num_step_episode += 1
                num_step_all += 1
                if not pos_now in pos_new_whole:
                    pos_new_whole.append(pos_now)
                if done:
                    break

            if len(in_conti_window) >= len_in_window:
                in_conti_window.pop(0)
            in_conti_window.append([fd_now,theta_now])

            state_next = rule_control.state_com(pos_now,fd_now,in_conti_window)
            reward = rule_control.reward_com(state_next,done)
            mask_done = 1.0 if done else 0.0
            
            example_experience = Experience(state_now, action, reward, state_next, mask_done)
            
            memory1.store_experience(example_experience)
            reward_all += reward
            
            if num_step_all > replay_train_size and save_model:
                agent_local.learn(memory1,batch_size1,gamma_D3QN,n_step)
                train_step += 1

            if done or num_step_episode >= max_episode_step or train_step % test_step_size == 0:
                print("{}Episode end!!!!!:done:{} ".format(i_episode,done))
                break
            
            if train_step >= 0 and epsilon > epsilon_min:
                epsilon = 1-(1/final_explore_step)*train_step
            
            get_local = False
            if get_out_local:
                len_pos_new = len(pos_new_whole)
                list_len_pos_new.append(len_pos_new)
                count = np.bincount(list_len_pos_new)
                if len(count) > 0:
                    if count[-1] >= max_local_step and epsilon < 1:
                        get_local = True
                        print("get local !!!!!!")


            while (get_local and get_out_local):
                area_index, pos_noreach = rule_control.global_state(pos_new_whole)
                print("area_index:{}".format(area_index))

                pos_local_goal = []
                while not done and num_step_episode <= max_episode_step:
                    action, num_conti_escape, pos_local_goal = rule_control.action_com(area_index, pos_noreach, pos_now,pos_local_goal)
                    
                    for i in range(num_conti_escape):
                        pos_now,fd_now,theta_now,done = env.step(action)
                        if not pos_now in pos_new_whole:
                            pos_new_whole.append(pos_now)

                        if done or num_step_episode > max_episode_step  or train_step % test_step_size == 0:
                            break
                        
                        if len(in_conti_window) >= len_in_window:
                            in_conti_window.pop(0)
                        in_conti_window.append([fd_now,theta_now])
                    
                    if pos_local_goal == pos_now or train_step % test_step_size == 0 or done:
                        break

                state_next = rule_control.state_com(pos_now,fd_now,in_conti_window)
                if done or num_step_episode > max_episode_step or train_step % test_step_size == 0:
                    break
                get_local = False
            state_now = state_next

        print("i_episode:{},num_step_all:{}, num_step_episode:{},train_step:{}, epsilon{}, reward_all:{},done:{} over---------".format(i_episode,num_step_all,num_step_episode, train_step,epsilon,reward_all,done))
        writer.add_scalar('train/reward', reward_all, num_step_all)
        writer.add_scalar('train/epsilon', epsilon, num_step_all)
        writer.close()
        
        if save_model and  i_episode % 1000 == 0 or num_step_all == max_step_all:
            save_agent_path = "agent1_model/d3qt_net1_{}_{}.pt".format(i_episode,train_step)
            agent_local.save_load_checkpoint("save",save_agent_path)
        
        if num_step_all > replay_train_size and train_step % test_step_size == 0:
            if train_step == 0:
                global mean_reward_list
                mean_reward_list = []
            mean_reward_list1 = testrun(agent_local, i_episode, train_step, mean_reward_list)
            mean_reward_list = mean_reward_list1

        i_episode += 1

    save_agent_path = "agent1_model/d3qt_net1_{}_{}.pt".format(i_episode,train_step)
    agent_local.save_load_checkpoint("save",save_agent_path)

def testrun(agent_local, i_episode, train_step, mean_reward_100_list):
    epsilon_test = 0
    step_episode_test = 1
    episode_reward_test = 0
    max_episode_step_test = 1300
    max_per_episode_step = 350
    while step_episode_test <= max_episode_step_test:
        done = False
        get_local = False
        while True:
            goal_name = goal_set[np.random.choice(len(goal_set))]
            start_pos = [random.randrange(4,n_w-5,2), random.randrange(4,n_w-5,2)]
            pos_now, fd_now, theta_now, goal_pos, fd_goal, done = env.reset(start_pos,goal_name)

            fd_conti_window = [fd_now]
            pos_new_whole = [pos_now]
            list_len_pos_new = []

            if not done:
                action = 0
                pos_now,fd_now,theta_now,done = env.step(action)
                in_conti_window = [[fd_now,theta_now]]
                pos_new_whole.append(pos_now)
                state_now = rule_control.state_com(pos_now,fd_now,in_conti_window)

            if not done:
                break

        d_s2g,_ = rule_control.distance(start_pos, goal_pos)
        max_local_step_test = 12
        step_episode_test1 = 1
        
        while not done and step_episode_test1 <= max_per_episode_step and step_episode_test <= max_episode_step_test:
            num_conti = 1
            action = agent_local.choose_action(state_now, epsilon_test)
            for i in range(num_conti):
                pos_now,fd_now,theta_now,done = env.step(action)
                step_episode_test += 1
                step_episode_test1 += 1
                if not pos_now in pos_new_whole:
                    pos_new_whole.append(pos_now)

                if done:
                    print("done:fd_conti_window:{}".format(fd_conti_window))
                    break

            if len(in_conti_window) >= len_in_window:
                in_conti_window.pop(0)
            in_conti_window.append([fd_now,theta_now])

            state_next = rule_control.state_com(pos_now,fd_now,in_conti_window)

            if done or step_episode_test > max_episode_step_test or step_episode_test1 > max_per_episode_step:
                break

            get_local = False
            if get_out_local:
                len_pos_new = len(pos_new_whole)
                list_len_pos_new.append(len_pos_new)
                count = np.bincount(list_len_pos_new)
                if len(count) > 0:
                    if count[-1] >= max_local_step_test:
                        get_local = True

            while (get_local and get_out_local):
                area_index, pos_noreach = rule_control.global_state(pos_new_whole)
                print("area_index:{}".format(area_index))

                pos_local_goal = []
                while not done and step_episode_test1 <= max_per_episode_step and step_episode_test <= max_episode_step_test:
                    action, num_conti_escape, pos_local_goal = rule_control.action_com(area_index, pos_noreach, pos_now,pos_local_goal)
                    
                    for i in range(num_conti_escape):
                        pos_now,fd_now,theta_now,done = env.step(action)
                        step_episode_test += 1
                        step_episode_test1 += 1
                        if not pos_now in pos_new_whole:
                            pos_new_whole.append(pos_now)
                        
                        if done or step_episode_test > max_episode_step_test or step_episode_test1 > max_per_episode_step:
                            break
                        
                        if len(in_conti_window) >= len_in_window:
                            in_conti_window.pop(0)
                        in_conti_window.append([fd_now,theta_now])
                    
                    if pos_local_goal == pos_now:
                        break

                state_next = rule_control.state_com(pos_now,fd_now,in_conti_window)
                get_local = False
                if done or step_episode_test > max_episode_step_test:
                    break
            state_now = state_next

        if done:
            episode_reward_test += 10

    if len(mean_reward_100_list) >= 50:
        mean_reward_100_list.pop(0)
    mean_reward_100_list.append(episode_reward_test)
    mean_reward_100 = np.mean(mean_reward_100_list)
    print("step_episode_test:{},episode_reward_test:{}".format(step_episode_test,episode_reward_test))

    if i_episode >= 0:
        all_list = [i_episode,train_step,episode_reward_test, mean_reward_100]
        pd_data = pd.DataFrame([all_list])
        pd_data.to_csv(df_path,mode='a',header=False,index=False)

    return mean_reward_100_list.copy()


if __name__ == "__main__":
    for i in range(1):
        df_path = "train_result/train_reward{}_{}_{}_{}_{}.csv".format(datetime.datetime.now().month,datetime.datetime.now().day,datetime.datetime.now().hour,datetime.datetime.now().minute,i)
        df = pd.DataFrame(columns=['episode','train_step','per_episode_reward','Mean episode cumulative reward'])
        df.to_csv(df_path,index=False)
        run(i)



