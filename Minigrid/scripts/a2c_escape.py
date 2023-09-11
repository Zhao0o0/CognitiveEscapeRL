import numpy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn
from .dictlist import DictList
from torch_ac.algos.base import BaseAlgo
from copy import deepcopy
from .welford import OnlineVariance
from .action_stats_logger import ActionStatsLogger
from .icm import ICM
import math
from .conversion_utils import scale_for_autoencoder
from utils.noisy_tv_wrapper import NoisyTVWrapper
#from position_memory import Position_memory
import random
from collections import deque

#position_memory = Position_memory()

class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(
        self,
        envs,
        acmodel,
        autoencoder,
        autoencoder_opt,
        uncertainty,
        noisy_tv,
        curiosity,
        randomise_env,
        uncertainty_budget,
        environment_seed,
        reward_weighting,
        normalise_rewards,
        frames_before_reset,
        device=None,
        num_frames_per_proc=None,
        discount=0.99,
        lr=0.01,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        rmsprop_alpha=0.99,
        rmsprop_eps=1e-8,
        preprocess_obss=None,
        reshape_reward=None,
        random_action=False,
    ):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(
            envs,
            acmodel,
            device,
            num_frames_per_proc,
            discount,
            lr,
            gae_lambda,
            entropy_coef,
            value_loss_coef,
            max_grad_norm,
            recurrence,
            preprocess_obss,
            reshape_reward,
        )

        self.optimizer = torch.optim.RMSprop(
            self.acmodel.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps
        )
        self.icm = ICM(
            autoencoder,
            autoencoder_opt,
            uncertainty,
            device,
            self.preprocess_obss,
            reward_weighting,
        )
        self.action = torch.Tensor([4] * 16)
        self.noisy_action_count = 0
        self.noisy_tv = noisy_tv
        self.curiosity = curiosity
        self.randomise_env = randomise_env
        self.uncertainty_budget = float(uncertainty_budget)
        self.environment_seed = int(environment_seed)
        self.visitation_counts = np.zeros(
            (self.env.envs[0].width, self.env.envs[0].height)
        )
        self.visitation_record_map_list = [np.zeros((self.env.envs[0].width, self.env.envs[0].height)) for _ in range(16)]
        shape = (self.num_frames_per_proc, self.num_procs)
        print('a2c.py:self.num_procs:{}'.format(self.num_procs))
        self.intrinsic_rewards = torch.zeros(*shape, device=self.device)
        self.uncertainties = torch.zeros(*shape, device=self.device)
        self.novel_states_visited = torch.zeros(*shape, device=self.device)
        self.reward_weighting = reward_weighting
        self.moving_average_calculator = OnlineVariance(device=self.device)
        self.moving_average_reward = OnlineVariance(device=self.device)
        self.normalise_rewards = normalise_rewards
        self.algo_count = 0
        self.frames_before_reset = int(frames_before_reset)
        self.saving_frames = False
        self.current_frames = []
        self.previous_frames = []
        self.predicted_frames = []
        self.predicted_uncertainty_frames = []
        self.intrinsic_reward_buffer = []
        self.agents_to_save = []
        self.counts_for_each_thread = [0] * 16
        self.action_stats_logger = ActionStatsLogger(self.env.envs[0].action_space.n)
        self.env = NoisyTVWrapper(self.env, self.noisy_tv)
        #print('a2c.py:109',type(self.env)) 
        #print('a2c.py:109', self.env.observation_space) 
        self.counter = 0
        self.random_action = random_action
        #self.local_optimum = False   #是否陷入局部最优
        self.max_deque_length = 6
        self.num_visited_positions_history = deque(maxlen=self.max_deque_length)
        self.num_visited_positions = []
        self.local_optimum = [False]*16
        self.num_differences = [10]*16
        self.num_visited_position_list = [0]*16
    '''
    def update_visitation_counts(self, envs):
        """
        updates counts of novel states visited
        """
        for i, env in enumerate(envs):
            if self.visitation_counts[env.agent_pos[0]][env.agent_pos[1]] == 0:
                pass
                #self.agents_to_save.append(i)
            self.visitation_counts[env.agent_pos[0]][env.agent_pos[1]] += 1
    '''
    def update_visitation_counts(self, positions):
        """
        updates counts of novel states visited
        """
        for i, position in enumerate(positions):
            if self.visitation_counts[position[0]][position[1]] == 0:
                pass
                #self.agents_to_save.append(i)
            self.visitation_counts[position[0]][position[1]] += 1

    # def record_visitation_nums(self, envs):
    #     """
    #     records counts of novel states visited
    #     """
    #     for i, env in enumerate(envs):
    #         if self.visitation_record_map_list[i][env.agent_pos[0]][env.agent_pos[1]] == 0:
    #             pass
    #             #self.agents_to_save.append(i)
    #         self.visitation_record_map_list[i][env.agent_pos[0]][env.agent_pos[1]] += 1
    
    def record_visitation_nums(self, positions):
        """
        records counts of novel states visited
        """
        for i, position in enumerate(positions):
            if self.visitation_record_map_list[i][position[0]][position[1]] == 0:
                pass
                #self.agents_to_save.append(i)
            self.visitation_record_map_list[i][position[0]][position[1]] += 1

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        # 16 threads running in parallel for 8 frames at a time before parameters
        # are updated, so gathers a total 128 frames
        loss = 0
        count = 0
        self.counter += 1
        for i in range(self.num_frames_per_proc):
            self.algo_count += 1
            self.counts_for_each_thread[i] += 1
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(
                        preprocessed_obs, self.memory * self.mask.unsqueeze(1)
                    )
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action_agent = dist.sample()
            if self.random_action == "True":
                action_used = np.random.randint(6, size=16)    #列表长度16
            elif self.random_action == "False":
                action_used = action_agent.cpu().numpy()
            else:
                raise ValueError("random_action must be True or False")
            
            #print('self.local_optimum',self.local_optimum)
            #判断是否局部最优
            escape_indices = [i_env for i_env, value in enumerate(self.local_optimum) if value]
            for i_env in escape_indices:
                print('self.local_optimum',self.local_optimum)
                print('before: i_env:{},action_used[i_env]:{}'.format(i_env, action_used[i_env]))
                action_escape = np.random.randint(3)
                action_used[i_env] = action_escape
                print('after: i_env:{},action_used[i_env]:{}'.format(i_env, action_used[i_env]))
                self.local_optimum[i_env] = False
            
            '''
            obs, extrinsic_reward, done, truncated, info = self.env.step(action)
            reward = extrinsic_reward
            '''
            obs, extrinsic_reward, terminated, truncated, info = self.env.step(action_used)
            done = tuple(value1 or value2 for value1, value2 in zip(terminated, truncated))            
            print('done',done)
            reward = extrinsic_reward
            
            #如果某个环境的回合终止，就重置这个环境的统计数目
            #if have done:
            # if True in done:
            #     index = done.index(True)           #########只更改了第一个环境的数据！！！！！！！！！！！！
            #     print("True_position:", index)
            #     self.visitation_record_map_list[index] = np.zeros(
            #         (self.env.envs[0].width, self.env.envs[0].height)
            #     )
            #     print('self.visitation_record_map_list[index]:',np.count_nonzero(self.visitation_record_map_list[index]))
            
            if True in done:
                #index = done.index(True)           #########只更改了第一个环境的数据！！！！！！！！！！！！
                for  index, value in enumerate(done):
                    if value:
                        self.visitation_record_map_list[index] = np.zeros(
                            (self.env.envs[0].width, self.env.envs[0].height)
                        )
                        print('self.visitation_record_map_list[index]:',np.count_nonzero(self.visitation_record_map_list[index]))
                '''
                print("True_position:", index)
                self.visitation_record_map_list[index] = np.zeros(
                    (self.env.envs[0].width, self.env.envs[0].height)
                )
                print('self.visitation_record_map_list[index]:',np.count_nonzero(self.visitation_record_map_list[index]))
                '''
            
            self.record_visitation_nums(self.env.envs)
            
            #self.num_visited_position = np.count_nonzero(self.visitation_record_num_list)   #列表(每个环境中已经被探索过的位置)
            for list_i in range(len(self.visitation_record_map_list)):
                
                self.num_visited_position_list[list_i] = np.count_nonzero(self.visitation_record_map_list[list_i])
                
            self.num_visited_positions_history.append(self.num_visited_position_list.copy())    #是队列，所以可以用append，一直保持队列长度固定
            
            if len(self.num_visited_positions_history) >= self.max_deque_length:
                first_list = self.num_visited_positions_history[0]
                last_list = self.num_visited_positions_history[-1]
                for first_list_i in range(len(self.env)):
                    first = first_list[first_list_i]
                    last = last_list[first_list_i]
                    num_difference = last - first
                    #self.num_differences[n_env] = num_difference
                    
                    if num_difference < 2:         #表示一直保持在一个位置
                        self.local_optimum[first_list_i] = True

            self.update_visitation_counts(self.env.envs)
            self.obss[i] = self.obs
            self.obs = obs
            #self.current_frames.append(self.obs)
            #self.previous_frames.append(self.obss[i])
            action = torch.from_numpy(action_used)
            
            if self.curiosity == "True":

                mse, intrinsic_reward, uncertainty = self.icm.compute_intrinsic_rewards(
                    self.obss[i], self.obs, action
                )
                if self.normalise_rewards == "True":
                    normlalised_reward = self.moving_average_reward.include_tensor(
                        intrinsic_reward
                    )
                    intrinsic_reward = normlalised_reward
                reward = intrinsic_reward + torch.tensor(reward, dtype=torch.float).to(
                    self.device
                )
                loss = torch.sum(mse)
                self.intrinsic_reward_buffer.append(intrinsic_reward)
                self.action_stats_logger.add_to_log_dicts(
                    action.detach().numpy(), intrinsic_reward.detach().numpy()
                )
                self.icm.update_curiosity_parameters(loss)

            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.curiosity == "True":
                self.uncertainties[i] = uncertainty
                self.intrinsic_rewards[i] = intrinsic_reward
            else:
                self.uncertainties[i] = torch.zeros_like(action)
                self.intrinsic_rewards[i] = torch.zeros_like(action)
            self.novel_states_visited[i] = np.count_nonzero(self.visitation_counts)
            if self.reshape_reward is not None:
                import pdb

                pdb.set_trace()
                self.rewards[i] = torch.tensor(
                    [
                        self.reshape_reward(obs_, action_, reward_, done_)
                        for obs_, action_, reward_, done_ in zip(
                            obs, action, reward, done
                        )
                    ],
                    device=self.device,
                )
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values
            self.log_episode_return += torch.tensor(
                reward, device=self.device, dtype=torch.float
            )
            self.log_episode_reshaped_return += self.rewards[i]
            
            
            self.log_episode_num_frames += torch.ones(
                self.num_procs, device=self.device
            )

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(
                        self.log_episode_reshaped_return[i].item()
                    )
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(
                    preprocessed_obs, self.memory * self.mask.unsqueeze(1)
                )
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = (
                self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            )
            next_value = (
                self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            )
            next_advantage = (
                self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0
            )

            delta = (
                self.rewards[i]
                + self.discount * next_value * next_mask
                - self.values[i]
            )
            self.advantages[i] = (
                delta + self.discount * self.gae_lambda * next_advantage * next_mask
            )

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [
            self.obss[i][j]
            for j in range(self.num_procs)
            for i in range(self.num_frames_per_proc)
        ]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(
                -1, *self.memories.shape[2:]
            )
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        intrinsic_rewards = self.intrinsic_rewards.transpose(0, 1).reshape(-1)
        uncertainties = self.uncertainties.transpose(0, 1).reshape(-1)
        novel_states_visited = self.novel_states_visited.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "uncertainties": uncertainties,
            "intrinsic_rewards": intrinsic_rewards,
            "novel_states_visited": novel_states_visited,
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs :]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs :]
        self.log_num_frames = self.log_num_frames[-self.num_procs :]

        return exps, logs

    def update_parameters(self, exps):
        """
        updates actor critic model parameters
        """
        # print("noisy action count", self.noisy_action_count)
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory
        print('self.acmodel.recurrent',self.acmodel.recurrent)
        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss

            if self.acmodel.recurrent:
                dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
            else:
                dist, value = self.acmodel(sb.obs)

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            loss = (
                policy_loss
                - self.entropy_coef * entropy
                + self.value_loss_coef * value_loss
            )

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = (
            sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        )
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm,
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
