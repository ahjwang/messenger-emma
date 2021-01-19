'''
Script for evaluating trained model performance from the command-line
'''
import argparse

import gym
import torch

from messenger.models.emma import EMMA
from messenger.models.utils import ObservationBuffer

def win_episode(model, env, args):
    '''
    Run the model on env for one episode and return True if the model won,
    return False otherwise.
    '''
    buffer = ObservationBuffer(buffer_size=3, device=args.device)
    obs, manual = env.reset()
    buffer.reset(obs)

    for t in range(args.max_steps):
        with torch.no_grad():
            action = model(buffer.get_obs(), manual)
        obs, reward, done, _ = env.step(action)
        if reward == 1: # get reward of 1 only if you win the game
            return True
        if done:
            break
        buffer.update(obs)
    return False

def total_wins(model, env, args):
    '''
    Run the models on the env and returns number of wins.
    '''
    wins = 0
    for _ in range(args.eval_eps):
        if win_episode(model, env, args):
            wins += 1
    return wins


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("--model_state", required=True, type=str, help="Path to model states to evaluate.")
    parser.add_argument("--eval_eps", default=100, type=int, help="Number of episodes to evaluate each model.")

    # environment arguments
    parser.add_argument("--env_id", required=True, type=str, help="Environment id used in gym.make")
    parser.add_argument("--max_steps", default=128, type=int, help="Maximum number of steps for each episode")

    args = parser.parse_args()

    # set the device
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    model = EMMA().to(args.device)
    model.load_state_dict(torch.load(args.model_state, map_location=args.device))
    model.eval()

    env = gym.make(args.env_id)

    # evaluate the models
    wins = total_wins(model, env, args)
    print(f"\nWin Rate: {wins/args.eval_eps:.3f} ({wins} / {args.eval_eps})\n")
