import gym
import deepq
import wandb


def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    env = gym.make("CarRacing-v0")
    deepq.learn(env, _wandb=1, model_identifier='agent')
    env.close()


if __name__ == '__main__':
    main()

