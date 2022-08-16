'''
Script for training models on stage 1, where the agent either starts with or with the goal
and the sole objective is to interact with the correct item.
'''

import argparse
import time
import pickle
import random
import hashlib

import gym
import messenger # this needs to be imported even though its not used to register gym environment ids
from messenger.models.utils import Encoder
import torch
from transformers import AutoModel, AutoTokenizer
import wandb
import numpy as np

from model import TrainEMMA, Memory
from train_tools import ObservationBuffer, PPO, TrainStats


def wrap_obs(obs):
    """ Convert obs format returned by gym env (dict) to a numpy array expected by model
    """
    return np.concatenate(
        (obs["entities"], obs["avatar"]), axis=-1
    )


def train(args):
    model_kwargs = {
        "hist_len": args.hist_len,
        "n_latent_var": args.latent_vars,
        "emb_dim": args.emb_dim,
    }

    optim_kwargs = {
        "weight_decay": args.weight_decay
    }

    ppo = PPO(
        ModelCls = TrainEMMA,
        model_kwargs = model_kwargs,
        device = args.device,
        lr = args.lr,
        gamma = args.gamma,
        K_epochs = args.k_epochs,
        eps_clip = args.eps_clip,
        load_state = args.load_state,
        optim_kwargs=optim_kwargs,
        optimizer=args.optimizer
    )

    # freeze attention mechanism
    if args.freeze_attention:
        ppo.policy.freeze_attention()
        ppo.policy_old.freeze_attention()

    # memory stores all the information needed by PPO to compute losses and make updates
    memory = Memory()

    # logging variables
    teststats = []
    runstats = []
        
    # make the environments
    env = gym.make(f'msgr-train-v{args.stage}')
    eval_env = gym.make(f'msgr-val-v{args.stage}')

    # training stat tracker
    eval_stats = TrainStats({-1: 'val_death', 1: "val_win"})
    train_stats = TrainStats({-1: 'death', 1: "win"})

    # Text Encoder
    encoder_model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(model=encoder_model, tokenizer=tokenizer, device=args.device, max_length=36)

    # Observation Buffer
    buffer = ObservationBuffer(device=args.device, buffer_size=args.hist_len)

    # training variables
    i_episode = 0
    timestep = 0
    max_win = -1
    max_train_win = -1
    start_time = time.time()

    while True: # main training loop
        obs, text = env.reset()
        obs = wrap_obs(obs)
        text = encoder.encode(text)
        buffer.reset(obs)

        # Episode loop
        for t in range(args.max_steps):
            timestep += 1

            # Running policy_old:
            action = ppo.policy_old.act(buffer.get_obs(), text, memory)
            obs, reward, done, _ = env.step(action)
            obs = wrap_obs(obs)
            
            # add the step penalty
            reward -= abs(args.step_penalty)

            # add rewards to memory and stats
            if t == args.max_steps - 1 and reward != 1:
                reward = -1.0 # failed to complete objective
                done = True
                
            train_stats.step(reward)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update the model if its time
            if timestep % args.update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
                
            if done:
                break
                
            buffer.update(obs)

        train_stats.end_of_episode()
        i_episode += 1

        # check if max_time has elapsed
        if time.time() - start_time > 60 * 60 * args.max_time:
            break

        # logging
        if i_episode % args.log_interval == 0:
            print("Episode {} \t {}".format(i_episode, train_stats))
            runstats.append(train_stats.compress())
            if not args.check_script:
                wandb.log(runstats[-1])
            
            if train_stats.compress()['win'] > max_train_win:
                torch.save(ppo.policy_old.state_dict(), args.output + "_maxtrain.pth")
                max_train_win = train_stats.compress()['win']
                
            train_stats.reset()

        # run evaluation
        if i_episode % args.eval_interval == 0:
            eval_stats.reset()
            ppo.policy_old.eval()

            for _ in range(args.eval_eps):
                obs, text = eval_env.reset()
                obs = wrap_obs(obs)
                text = encoder.encode(text)
                buffer.reset(obs)

                # Running policy_old:
                for t in range(args.max_steps):
                    with torch.no_grad():
                        action = ppo.policy_old.act(buffer.get_obs(), text, None)
                    obs, reward, done, _ = eval_env.step(action)
                    obs = wrap_obs(obs)
                    if t == args.max_steps - 1 and reward != 1:
                        reward = -1.0 # failed to complete objective
                        done = True
                        
                    eval_stats.step(reward)
                    if done:
                        break
                    buffer.update(obs)
                eval_stats.end_of_episode()

            ppo.policy_old.train()

            print("TEST: \t {}".format(eval_stats))
            teststats.append(eval_stats.compress(append={"step": train_stats.total_steps}))
            if not args.check_script:
                wandb.log(teststats[-1])

            if eval_stats.compress()['val_win'] > max_win:
                torch.save(ppo.policy_old.state_dict(), args.output + "_max.pth")
                max_win = eval_stats.compress()['val_win']
                
            # Save metrics
            with open(args.output + "_metrics.pkl", "wb") as file:
                pickle.dump({"test": teststats, "run": runstats}, file)

            # Save model states
            torch.save(ppo.policy_old.state_dict(), args.output + "_state.pth")
            
        if i_episode > args.max_eps:
            break
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--output", required=True, type=str, help="Local output file name or path.")
    parser.add_argument("--seed", default=None, type=int, help="Set the seed for the model and training.")
    parser.add_argument("--device", default=0, type=int, help="cuda device ordinal to train on.")

    # Model arguments
    parser.add_argument("--load_state", default=None, help="Path to model state dict.")
    parser.add_argument("--latent_vars", default=128, type=int, help="Latent model dimension.")
    parser.add_argument("--hist_len", default=3, type=int, help="Length of history used by state buffer")
    parser.add_argument("--emb_dim", default=256, type=int, help="embedding size for text")

    # Environment arguments
    parser.add_argument("--stage", default=1, type=int, help="the stage to run experiment on")
    parser.add_argument("--max_steps", default=4, type=int, help="Maximum num of steps per episode")
    parser.add_argument("--step_penalty", default=0.0, type=float, help="negative reward for each step")
    
    # Training arguments
    parser.add_argument("--update_timestep", default=64, type=int, help="Number of steps before model update")
    parser.add_argument("--lr", default=0.00005, type=float, help="learning rate")
    parser.add_argument("--gamma", default=0.8, type=float, help="discount factor")
    parser.add_argument("--k_epochs", default=4, type=int, help="num epochs to update")
    parser.add_argument("--eps_clip", default=0.1, type=float, help="clip param for PPO")
    parser.add_argument("--optimizer", default="Adam", type=str, help="optimizer class to use")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay for optimizer")
    parser.add_argument("--max_time", default=1000, type=float, help="max train time in hrs")
    parser.add_argument("--max_eps", default=1e10, type=float, help="max training episodes")
    parser.add_argument("--freeze_attention", action="store_true", help="Do not update attention weights.")

    # Logging arguments
    parser.add_argument('--log_interval', default=5000, type=int, help='number of episodes between logging')
    parser.add_argument('--eval_interval', default=25000, type=int, help='number of episodes between eval')
    parser.add_argument('--eval_eps', default=500, type=int, help='number of episodes to run eval')
    parser.add_argument('--log_group', type=str, help="wandb log group")
    parser.add_argument('--entity', type=str, help="entity to log runs to on wandb")
    parser.add_argument('--check_script', action='store_true', help="run quickly just to see script runs okay.")

    args = parser.parse_args()
    
    # get hash of arguments minus seed
    args_dict = vars(args).copy()
    args_dict["device"] = None
    args_dict["seed"] = None
    args_dict["output"] = None
    args_hash = hashlib.md5(
        str(sorted(args_dict.items())).encode("utf-8")
    ).hexdigest()

    args.device = torch.device(f"cuda:{args.device}")

    if args.seed is not None: # seed everything
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    if args.check_script: # for quick debugging
        args.eval_interval = 50
        args.eval_eps = 50
        args.log_interval = 50
        args.max_eps = 100

    else:
        wandb.init(
            project = "msgr-emma",
            entity = args.entity,
            group = args.log_group if args.log_group is not None else args_hash,
            name = f"emma_stage-{args.stage}_seed-{args.seed}"
        )
        wandb.config.update(args)
    
    train(args)