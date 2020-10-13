import agent
import tensorflow as tf
import cv2
from tf_agents.environments import tf_py_environment
import imageio

eval_env = tf_py_environment.TFPyEnvironment(agent.Game(limit=False))

name = 'perfect_policies/policy_5_width_max_score'
saved_policy = tf.compat.v2.saved_model.load(name)
time_step = eval_env.reset()

num_frames = 0
fps = 24

with imageio.get_writer(name + ".mp4", fps=fps) as video:
    while num_frames < 10000:
        policy_step = saved_policy.action(time_step)
        time_step = eval_env.step(policy_step.action)
        video.append_data(cv2.resize(eval_env.render()[0].numpy(), (320, 400), interpolation=cv2.INTER_NEAREST))
        num_frames += 1
