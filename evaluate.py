'''
Script for evaluating trained model performance from the command-line
'''

import argparse

import numpy as np
import torch

# Environment
import gym
import messenger

# models
from messenger.models.emma import EMMA
from messenger.models.utils import ObservationBuffer


def run_episode(model, env, args):
    '''
    Run the model on env for one episode and return True if the model won,
    return False otherwise.
    '''
    
    buffer = ObservationBuffer(buffer_size=3, device=args.device)
    obs, manual = env.reset()
    
    buffer.reset(obs)
    total_reward = 0
    
    for t in range(args.max_steps):
        with torch.no_grad():
            action = model(buffer.get_obs(), manual)
        obs, reward, done, _ = env.step(action)
        
        total_reward += reward
        
        if done and reward == 1:
            return True, total_reward
        
        if t == args.max_steps - 1 and not done:
            return False, total_reward - 1
            
        if done:
            break
            
        buffer.update(obs)
        
    return False, total_reward


def evaluate_model(model, env, args):
    '''
    Run the models on the env and report mean and std win rates.
    '''
    win_rates = []
    avg_rewards = []
    for i, state in enumerate(args.model_states):
        model.load_state_dict(torch.load(state, map_location=args.device))
        model.eval()
        wins = 0
        rewards = 0
        for _ in range(args.eval_eps):
            win, reward = run_episode(model, env, args)
            if win:
                wins += 1
            rewards += reward
            
        win_rates.append(wins / args.eval_eps)
        avg_rewards.append(rewards / args.eval_eps)

    return np.mean(win_rates), np.std(win_rates), np.mean(avg_rewards), np.std(avg_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("--model_states", required=True, nargs="+", type=str, help="Path to model states to evaluate.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device ordinal.")
    parser.add_argument("--eval_eps", default=1000, type=int, help="Number of episodes to evaluate each model.")

    # environment arguments
    parser.add_argument("--env_id", required=True, type=str, help="gym env id")
    parser.add_argument("--max_steps", default=256, type=int, help="max number of steps in each episode.")
        
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device}")

    model = EMMA().to(args.device)
    
    # Make the environment
    env = gym.make(args.env_id)

    # evaluate the models
    mean_win, std_win, mean_rew, std_rew = evaluate_model(model, env, args)
    print(f"\nWin Rate: {mean_win:.3f} \u00B1 {std_win:.3f}\n")
    print(f"Rewards:  {mean_rew:.3f} \u00B1 {std_rew:.3f}\n")