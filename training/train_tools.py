import torch
import torch.nn as nn


class ObservationBuffer:
    '''
    Maintains a buffer of observations along the 0-dim.
    
    Parameters:
    buffer_size
        How many previous observations to track in the buffer
    device
        The device on which buffers are loaded into

    '''
    def __init__(self, buffer_size, device,):
        self.buffer_size = buffer_size
        self.buffer = None
        self.device = device

    def _np_to_tensor(self, obs):
        return torch.from_numpy(obs).long().to(self.device)

    def reset(self, obs):
        # initialize / reset the buffer with the observation
        self.buffer = [self._np_to_tensor(obs) for _ in range(self.buffer_size)]

    def update(self, obs):
        # update the buffer by appending newest observation
        assert self.buffer, "Please initialize buffer first with reset()"
        del self.buffer[0] # delete the oldest entry
        self.buffer.append(self._np_to_tensor(obs)) # append the newest observation

    def get_obs(self):
        # get a stack of all observations currently in the buffer
        return torch.stack(self.buffer)


class PPO:
    '''
    Minimal PPO implementation based on https://github.com/nikhilbarhate99/PPO-PyTorch
    '''
    def __init__(self, ModelCls, model_kwargs, device, lr, gamma, K_epochs, eps_clip, load_state=None, load_optim=None,
                optim_kwargs = {}, optimizer="Adam"
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.MseLoss = nn.MSELoss()
        self.policy = ModelCls(**model_kwargs).to(device)
        self.policy_old = ModelCls(**model_kwargs).to(device)
        
        for p in self.policy.parameters(): # xavier initialize model weights
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if load_state: # initialize model weights if one is specified
            self.policy.load_state_dict(torch.load(load_state, map_location=device))

        self.policy_old.load_state_dict(self.policy.state_dict())

        OptimCls = getattr(torch.optim, optimizer)
        self.optimizer = OptimCls(
            self.policy.parameters(),
            lr=lr,
            **optim_kwargs
        )

        if load_optim:
            self.optimizer.load_state_dict(torch.load(load_optim, map_location=device))
        
        self.policy.train()
        self.policy_old.train()

    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        old_texts = torch.stack(memory.texts).to(self.device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_texts)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) \
                - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # zero out the padding_idx for sprite embedder (temp fix for PyTorch bug)
        self.policy.sprite_emb.weight.data[0] = 0
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class TrainStats:
    '''
    Class for tracking agent reward and other statistics.
    Params:
        rewards_key: a dictionary that maps from reward (float) to a string that describes the meaning of the reward. E.g. {1:'resource', 2:'key', 3:'win'}
    '''
    def __init__(self, rewards_key):
        self._reserved = ['all_rewards', 'avg_reward', 'avg_length', 'episodes']
        self._rk = rewards_key.copy()
        self.stats = {}
        for event in rewards_key.values():
            if event in self._reserved:
                raise Exception(event + ' is a reserved event word')
            self.stats[event] = 0
        self.all_rewards = [] # end of episode rewards
        self.eps_reward = 0 # track rewards in episode
        self.steps = 0 # steps since last reset()
        self.total_steps = 0 # steps since object instantiation
        self.episodes = 0
        
    # return str stats
    def __str__(self):
        if self.episodes == 0:
            return 'No stats'
        stats = f'lengths: {self.steps/self.episodes:.2f} \t '
        stats += f'rewards: {sum(self.all_rewards)/self.episodes:.2f} \t '
        for key, val in self.stats.items():
            stats += (key + 's: ' + f'{val/self.episodes:.2f}' + ' \t ')
        return stats
        
    # adds reward to current set of stats
    def step(self, reward):
        self.steps += 1
        self.total_steps += 1
        self.eps_reward += reward
        for key, event in self._rk.items():
            if reward == key:
                self.stats[event] += 1
    
    # end of episode
    def end_of_episode(self):
        self.episodes += 1
        self.all_rewards.append(self.eps_reward)
        self.eps_reward = 0
    
    # reset gamestats
    def reset(self):
        for key in self.stats.keys():
            self.stats[key] = 0
        self.episodes = 0
        self.all_rewards = []
        self.eps_reward = 0
        self.steps = 0
        self.episodes = 0

    # the running reward
    def running_reward(self):
        assert self.episodes > 0, 'number of episodes = 0'
        return sum(self.all_rewards) / self.episodes
        
    # compress all stats into single dict. Append is optional dict to append
    def compress(self, append=None):
        assert self.episodes > 0, 'No stats to compress'
        stats = {event: num/self.episodes for (event, num) in self.stats.items()}
        stats['step'] = self.total_steps
        stats['all_rewards'] = self.all_rewards.copy()
        stats['avg_reward'] = sum(self.all_rewards) / self.episodes
        stats['avg_length'] = self.steps / self.episodes
        stats['episodes'] = self.episodes
        if append:
            stats.update(append)
        return stats