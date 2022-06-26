#!/usr/bin/python

import sys
sys.path.append("../src")
import gym
import pybullet_envs as pe
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import wandb
from config import *
from replay_buffer import *
from networks import *
from agent import *
import imageio 
from IPython.display import Image


config = dict(
  learning_rate_actor = ACTOR_LR,
  learning_rate_critic = ACTOR_LR,
  batch_size = BATCH_SIZE,
  architecture = "SAC",
  env = ENV_NAME
)

wandb.init(
  project=f"tensorflow2_sac_{ENV_NAME.lower()}",
  tags=["SAC", "FCL", "RL"],
  config=config,
)

env = gym.make(ENV_NAME)
agent = Agent(env)

scores = []
evaluation = True
images = []


for _ in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    states = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.get_action(states)
        new_states, reward, done, info = env.step(action)
        score += reward
        agent.add_to_replay_buffer(states, action, reward, new_states, done)
        img = env.render(mode='rgb_array')
        images.append(img)
        states = new_states
        agent.learn()
        states = new_states
    
    
    scores.append(score)
    agent.replay_buffer.update_n_games()

    wandb.log({'Game number': agent.replay_buffer.n_games, '# Episodes': agent.replay_buffer.buffer_counter, 
               "Average reward": round(np.mean(scores[-10:]), 2), \
                      "Time taken": round(time.time() - start_time, 2)})
    
    if (_ + 1) % SAVE_FREQUENCY == 0:
        print("saving...")
        agent.save()
        print("saved")

imageio.mimsave('demo.mp4', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)








