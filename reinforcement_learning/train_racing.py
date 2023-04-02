import gym
import deepq


def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    env = gym.make("CarRacing-v0")
    deepq.learn(env, action_repeat=8, model_identifier='agent_action_repeat=8')
    env.close()


if __name__ == '__main__':
    main()

