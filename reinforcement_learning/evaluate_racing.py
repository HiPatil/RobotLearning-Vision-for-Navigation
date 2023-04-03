import gym
import deepq


def main():
    """ 
    Evaluate a trained Deep Q-Learning agent 
    """ 
    env = gym.make("CarRacing-v0")
    deepq.evaluate(env, load_path='models/agent_SmallModel_NormClip_DDQN.pt')
    env.close()

if __name__ == '__main__':
    main()
