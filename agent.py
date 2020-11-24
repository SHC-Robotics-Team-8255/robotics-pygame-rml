import pygame
import numpy as np
import time
import random
import imageio
import tensorflow as tf
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import policy_saver

tf.compat.v1.enable_v2_behavior()


class Field:

    def __init__(self, row_blocks, col_blocks, block_width, block_height):
        self.row_width = row_blocks
        self.col_height = col_blocks

        self.block_width = block_width
        self.block_height = block_height

        self.total_width = self.block_width * self.row_width
        self.total_height = self.block_height * self.col_height

        self.field = np.zeros((col_blocks, row_blocks), int)
        # self.field = [[0].copy() * row_blocks].copy() * col_blocks

        # self.field[1][1] = 1

        self.gate_width = 5

        self.padding = 2

        self.next_gate = True

        self.layers_per_gate = 2

        self.layers_left = self.layers_per_gate

        self.gate = self.generate_gate()

    def generate_gate(self):
        gate = np.ones(self.row_width)

        gate_left_start = random.randrange(self.padding, self.row_width - self.padding - self.gate_width)

        for i in range(gate_left_start, gate_left_start + self.gate_width):
            gate[i] = 0

        return gate

    def shorten_gate(self):
        self.gate_width = max(5, self.gate_width - 1)

    def update(self):
        return_true = False
        if self.field[19][0] == 1 and not self.next_gate:
            self.next_gate = True
            self.gate = self.generate_gate()
            return_true = True

        self.field = np.delete(self.field, 19, 0)
        if self.next_gate:
            self.field = np.insert(self.field, 0, self.gate, 0)
            self.layers_left -= 1
            if self.layers_left == 0:
                self.next_gate = False
                self.layers_left = self.layers_per_gate
        else:
            self.field = np.insert(self.field, 0, np.zeros(self.row_width, int), 0)

        return return_true

    def __repr__(self):
        return " " + self.field.__repr__().replace("],", "],\n ")


class Game(py_environment.PyEnvironment):
    block_width = 20
    block_height = 20

    row_blocks = 16
    col_blocks = 20

    def __init__(self, limit=True):

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(7, 16), dtype=np.int32, minimum=0, name='observation')

        self.field = Field(self.row_blocks, self.col_blocks, self.block_width, self.block_height)
        self.render_field = None

        # self.window = pygame.display.set_mode((self.field.total_width, self.field.total_height))

        # self.clock = pygame.time.Clock()

        self.game_over = False

        self.player = [7, 16]

        self.difficulty = 0
        self.difficulty_jump = 100

        self.score = 0

        self._step_count = 0
        self._episode_ended = False

        self.limit = limit

        # self.start_game_loop()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # self._state = 0
        self._step_count = 0
        self.field = Field(self.row_blocks, self.col_blocks, self.block_width, self.block_height)
        self._episode_ended = False
        self.render_field = self.field.field.copy()
        self.render_field[self.player[1]][self.player[0] % self.row_blocks] += 2
        self.render_field[self.player[1] + 1][self.player[0] % self.row_blocks] += 2
        self.render_field[self.player[1]][(self.player[0] + 1) % self.row_blocks] += 2
        self.render_field[self.player[1] + 1][(self.player[0] + 1) % self.row_blocks] += 2
        return ts.restart(self.render_field[11:18])

    # def increase_score(self):
    #    self.score += 1

    def start_game_loop(self):
        frame_count = 0
        while not self.game_over:

            if frame_count % self.difficulty_jump == 0:
                self.difficulty += 1
                self.field.shorten_gate()

            events = pygame.event.get()
            action = 0

            for event in events:
                if event.type == pygame.QUIT:
                    self.game_over = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    action -= 1
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    action += 1

            self.player[0] += action

            if frame_count % 4 == 0:
                self.field.update()
            self.render()
            frame_count += 1
        time.sleep(3)

    def _step(self, action):

        if self.limit and self._step_count > 10000:
            self._episode_ended = True
            reward = 100
            return ts.termination(self.render_field[11:18], reward)

        if self._episode_ended:
            return self.reset()

        if self._step_count % self.difficulty_jump == 0:
            self.difficulty += 1
            self.field.shorten_gate()

        # events = pygame.event.get()
        total_action = 0
        # print(action)

        if action == 0:
            total_action = -1
        else:
            total_action = 1

        # for event in events:
        #    if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
        #        action -= 1
        #    if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
        #        action += 1

        self.player[0] += total_action

        reward = 1.0
        if self._step_count % 3 == 0:
            self.field.update()
            #    reward = 10.0
        # self.render()

        self.render_field = self.field.field.copy()

        self.render_field[self.player[1]][self.player[0] % self.row_blocks] += 2

        self.render_field[self.player[1] + 1][self.player[0] % self.row_blocks] += 2

        self.render_field[self.player[1]][(self.player[0] + 1) % self.row_blocks] += 2

        self.render_field[self.player[1] + 1][(self.player[0] + 1) % self.row_blocks] += 2

        # print(self.render_field[12: 17])
        # self.render()

        for row in range(len(self.render_field)):
            for block in range(len(self.render_field[row])):
                if self.render_field[row][block] == 3:
                    self._episode_ended = True
                    reward = -100
                    return ts.termination(self.render_field[11:18], reward)

        self._step_count += 1
        return ts.transition(self.render_field[11:18], reward=reward, discount=1.0)

    def get_color(self, number):
        if np.equal(number, 0):
            return (255, 255, 255)
        elif np.equal(number, 1):
            return (0, 0, 0)
        elif np.equal(number, 2):
            return (255, 0, 0)
        else:
            self.game_over = True
            return (0, 255, 0)

    def render(self, mode=''):

        # self.window.fill((0, 0, 0))
        render = np.zeros((20, 16, 3))

        for row in range(len(self.render_field)):
            for block in range(len(self.render_field[row])):
                rectangle = pygame.Rect(block * self.block_width, row * self.block_height, self.block_width,
                                        self.block_height)
                # pygame.draw.rect(self.window, self.get_color(self.render_field[row][block]), rectangle)
                render[row][block] = list(self.get_color(self.render_field[row][block]))

        # self.clock.tick(24)
        # pygame.display.update()
        return render


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


if __name__ == '__main__':
    environment = Game()
    utils.validate_py_environment(environment, episodes=5)


    train_env = tf_py_environment.TFPyEnvironment(Game())
    eval_env = tf_py_environment.TFPyEnvironment(Game())

    num_iterations = 100000  # @param {type:"integer"}

    initial_collect_steps = 200  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 2000  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 5000  # @param {type:"integer"}

    fc_layer_params = (100,)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    print(compute_avg_return(eval_env, random_policy, num_eval_episodes))


    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)


    def collect_data(env, policy, buffer, steps):
        for _ in range(steps):
            collect_step(env, policy, buffer)


    collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
            policy_dir = os.path.join(os.getcwd(),
                                      f'policies/epoch_run/policy_{round(step)}')
            tf_policy_saver.save(policy_dir)


    def create_policy_eval_video(policy, filename, num_episodes=20, fps=30):
        with imageio.get_writer(filename, fps=fps) as video:
            for _ in range(num_episodes):
                time_step = eval_env.reset()
                video.append_data(cv2.resize(eval_env.render()[0].numpy(), (320, 400), interpolation=cv2.INTER_NEAREST))
                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    time_step = eval_env.step(action_step.action)
                    video.append_data(cv2.resize(eval_env.render()[0].numpy(), (320, 400), interpolation=cv2.INTER_NEAREST))
                    cv2.imshow('frame',
                               cv2.resize(eval_env.render()[0].numpy(), (320, 400), interpolation=cv2.INTER_NEAREST))
                    cv2.waitKey(1)


    print(compute_avg_return(eval_env, random_policy, num_eval_episodes))

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.show()

    create_policy_eval_video(agent.policy, "test.mp4")

    #policy_dir = os.path.join(os.getcwd(),
    #                          f'policies/epoch_run/policy_{round(compute_avg_return(eval_env, random_policy, num_eval_episodes), 2)}')
    #tf_policy_saver.save(policy_dir)
