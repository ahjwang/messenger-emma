'''
Script that allows users to play Messenger in the terminal.
'''

import argparse
import numpy as np
import gym
import messenger

def numpy_formatter(i: int):
    ''' Format function passed to numpy print to make things pretty.
    '''
    id_map = {}
    for ent in messenger.envs.config.ALL_ENTITIES:
        id_map[ent.id] = ent.name[:2].upper()
    id_map[0] = '  '
    id_map[15] = 'A0'
    id_map[16] = 'AM'
    if i < 17:
        return id_map[i]
    else:
        return 'XX'

def print_instructions():
    ''' Print the Messenger instructions and header.
    '''
    print(f"\nMESSENGER\n")
    print("Read the manual to get the message and bring it to the goal.")
    print("A0 is you (agent) without the message, and AM is you with the message.")
    print("The following is the symbol legend (symbol : entity)\n")
    for ent in messenger.envs.config.ALL_ENTITIES[:12]:
        print(f"{ent.name[:2].upper()} : {ent.name}")    
    print("\nNote when entities overlap the symbol might not make sense. Good luck!\n")

def print_grid(obs):
    ''' Print the observation to terminal
    '''
    grid = np.concatenate((obs['entities'], obs['avatar']), axis=-1)
    print(np.sum(grid, axis=-1).astype('uint8'))

def print_manual(manual):
    ''' Print the manual to terminal
    '''
    man_str = f"Manual: {manual[0]}\n"
    for description in manual[1:]:
        man_str += f"        {description}\n"
    print(man_str)

def clear_terminal():
    ''' Special print that will clear terminal after each step.
    Replace with empty return if your terminal has issues with this.
    ''' 
    # print(chr(27) + "[2J")
    print("\033c\033[3J")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True, help='environment id for human play')
    args = parser.parse_args()

    np.set_printoptions(formatter={'int': numpy_formatter})
    env = gym.make(args.env_id)

    # map from keyboard entry to gym action space
    action_map = {'w': 0, 's': 1, 'a': 2, 'd': 3, '': 4}
    
    keep_playing = "yes"
    total_games = 0
    total_wins = 0

    while keep_playing.lower() not in ["no", "n"]:
        obs, manual = env.reset()
        done = False
        eps_reward = 0
        eps_steps = 0
        reward = 0
        print_instructions()
        print_manual(manual)
        print_grid(obs)
        action = input('\nenter action [w,a,s,d,\'\']: ')

        while not done:
            if action.lower() in action_map:
                obs, reward, done, info = env.step(action_map[action])
                eps_steps += 1
                eps_reward += reward
                clear_terminal()
                print_instructions()
                print_manual(manual)
                print_grid(obs)

                if reward != 0:
                    print(f"\ngot reward: {reward}\n")
            if done:
                total_games += 1
                if reward == 1:
                    total_wins += 1
                    print("\n\tcongrats! you won!!\n")
                else:
                    print("\n\tyou lost :( better luck next time.\n")
                break

            action = input('\nenter action [w,a,s,d,\'\']: ')

        print(f"\nFinished episode with reward {eps_reward} in {eps_steps} steps!\n")
        keep_playing = input("play again? [n/no] to quit: ")
        if keep_playing.lower() not in ["no", "n"]:
            clear_terminal()

    print(f"\nThanks for playing! You won {total_wins} / {total_games} games.\n")