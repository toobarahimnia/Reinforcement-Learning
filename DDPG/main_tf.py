from DDPG import Agent
import gym
import numpy as np
# from utils import label_map_util
# from utils import visualization_utils as vis_util

from utils import plotLearning


if __name__=='__main__':
    env = gym.make('Pendulum-v1')
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[3], tau=0.001, env=env,
                    batch_size=64, layer1_size=800, layer2_size=600, n_actions=1)
    score_history = []
    np.random.seed(0)
    for i in range(1000):  
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act =agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score,
                        '100 games average %.2f' % np.mean(score_history[-100:]))

    filename = 'pendulum.png'
    plotLearning(score_history, filename, window=100)