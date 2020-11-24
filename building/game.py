import pygame
import numpy as np
import time
import random
import imageio
import tensorflow as tf
import os
import cv2

from tf_agents.specs import array_spec

class Field:

    def __init__(self, row_blocks, col_blocks, block_width, block_height):
        self.row_width = row_blocks
        self.col_height = col_blocks

        self.block_width = block_width
        self.block_height = block_height

        self.total_width = self.block_width * self.row_width
        self.total_height = self.block_height * self.col_height

        self.field = np.zeros((col_blocks, row_blocks), int)

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


class Game():
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

        self.game_over = False

        self.player = [7, 16]

        # self.difficulty = 0
        # self.difficulty_jump = 100

        self.score = 0

        self._step_count = 0
        self._episode_ended = False

        self.limit = limit


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._step_count = 0
        self.field = Field(self.row_blocks, self.col_blocks, self.block_width, self.block_height)
        self._episode_ended = False
        self.render_field = self.field.field.copy()
        self.render_field[self.player[1]][self.player[0] % self.row_blocks] += 2
        self.render_field[self.player[1] + 1][self.player[0] % self.row_blocks] += 2
        self.render_field[self.player[1]][(self.player[0] + 1) % self.row_blocks] += 2
        self.render_field[self.player[1] + 1][(self.player[0] + 1) % self.row_blocks] += 2
        # return ts.restart(self.render_field[11:18])

    def _step(self, action):

        if self.limit and self._step_count > 10000:
            self._episode_ended = True
            reward = 100
            # return ts.termination(self.render_field[11:18], reward)

        if self._episode_ended:
            return self._reset()

        #if self._step_count % self.difficulty_jump == 0:
        #    self.difficulty += 1
        #    self.field.shorten_gate()

        # events = pygame.event.get()
        total_action = 0
        # print(action)

        if action == 0:
            total_action = -1
        else:
            total_action = 1

        self.player[0] += total_action

        reward = 1.0
        if self._step_count % 3 == 0:
            self.field.update()

        self.render_field = self.field.field.copy()

        self.render_field[self.player[1]][self.player[0] % self.row_blocks] += 2

        self.render_field[self.player[1] + 1][self.player[0] % self.row_blocks] += 2

        self.render_field[self.player[1]][(self.player[0] + 1) % self.row_blocks] += 2

        self.render_field[self.player[1] + 1][(self.player[0] + 1) % self.row_blocks] += 2

        # print(self.render_field[12: 17])
        
        cv2.imshow("game", cv2.resize(self.render(), (320, 400), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(33)

        for row in range(len(self.render_field)):
            for block in range(len(self.render_field[row])):
                if self.render_field[row][block] == 3:
                    self._episode_ended = True
                    reward = -100
                    # return ts.termination(self.render_field[11:18], reward)

        self._step_count += 1
        # return ts.transition(self.render_field[11:18], reward=reward, discount=1.0)

    def get_color(self, number):
        if np.equal(number, 0):
            return (255, 255, 255)
        elif np.equal(number, 1):
            return (0, 0, 0)
        elif np.equal(number, 2):
            return (255, 0, 0)
        else:
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


game = Game()
for step in range(500):
    game._step(random.randrange(0, 2))