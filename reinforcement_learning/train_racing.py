import gym
import deepq


def main():
    """ 
    Train a Deep Q-Learning agent 
    """ 
    env = gym.make("CarRacing-v0")
    deepq.learn(env, model_identifier='agent_SmallModel_NormClip_dqn')
    env.close()


if __name__ == '__main__':
    main()

