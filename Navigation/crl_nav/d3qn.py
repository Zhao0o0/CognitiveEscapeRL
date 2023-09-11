import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle,os,time


LR = 0.001
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 20000

ENV_A_SHAPE = 0


class dueling_ddqn(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(dueling_ddqn, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc = nn.Linear(self.state_dim, 256)

        self.adv_fc1 = nn.Linear(256, 128)
        self.adv_fc2 = nn.Linear(128, self.action_dim)

        self.value_fc1 = nn.Linear(256, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, state):
        feature = self.fc(state)
        advantage = self.adv_fc2(F.relu(self.adv_fc1(F.relu(feature))))
        value = self.value_fc2(F.relu(self.value_fc1(F.relu(feature))))
        Q = advantage + value - advantage.mean()
        return Q


class D3QN_Nstep(object):
    def __init__(self,N_STATES,N_ACTIONS):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device:{}".format(self.device))
        self.eval_net = dueling_ddqn(N_STATES,N_ACTIONS).to(self.device)
        self.target_net = dueling_ddqn(N_STATES,N_ACTIONS).to(self.device)

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss_func = self.loss_func.to(self.device)

    def choose_action(self, x, epsilon):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        if np.random.uniform() > epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def learn(self,memory,batch_size,gamma,n_step):
        state, action, reward, next_state, done = memory.sample(batch_size)
        
        q_values = self.eval_net.forward(state)
        next_q_values = self.target_net.forward(next_state)
        argmax_actions = self.eval_net.forward(next_state).max(1)[1].detach()
        next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + (gamma ** n_step) * (1 - done) * next_q_value

        loss = self.loss_func(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1


    def save_load_checkpoint(self,op,ckpt_path):
        if op == "save":
            torch.save({'eval_net_dict': self.eval_net.state_dict()}, ckpt_path)

        if op == "load":
            if ckpt_path is not None:
                checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
                self.eval_net.load_state_dict(checkpoint['eval_net_dict'])
