import pygame
import numpy as np
import time
import random

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

        self.gate_width = 10

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
        self.gate_width = max(2, self.gate_width - 1)

    def update(self):
        if self.field[19][0] == 1 and not self.next_gate:
            self.next_gate = True
            self.gate = self.generate_gate()
        self.field = np.delete(self.field, 19, 0)
        if self.next_gate:
            self.field = np.insert(self.field, 0, self.gate, 0)
            self.layers_left -= 1
            if self.layers_left == 0:
                self.next_gate = False
                self.layers_left = self. layers_per_gate
        else:
            self.field = np.insert(self.field, 0, np.zeros(self.row_width, int), 0)

    def __repr__(self):
        return " " + self.field.__repr__().replace("],", "],\n ")


class Game:
    block_width = 20
    block_height = 20

    row_blocks = 16
    col_blocks = 20

    def __init__(self):
        self.field = Field(self.row_blocks, self.col_blocks, self.block_width, self.block_height)

        self.window = pygame.display.set_mode((self.field.total_width, self.field.total_height))

        self.clock = pygame.time.Clock()

        self.game_over = False

        self.player = [7, 16]

        self.start_game_loop()

    def start_game_loop(self):
        frame_count = 0
        while not self.game_over:
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

    def render(self):

        self.window.fill((0, 0, 0))

        field = self.field.field.copy()

        field[self.player[1]][self.player[0] % self.row_blocks] += 2

        field[self.player[1] + 1][self.player[0] % self.row_blocks] += 2

        field[self.player[1]][(self.player[0] + 1) % self.row_blocks] += 2

        field[self.player[1] + 1][(self.player[0] + 1) % self.row_blocks] += 2

        for row in range(len(field)):
            for block in range(len(field[row])):
                rectangle = pygame.Rect(block * self.block_width, row * self.block_height, self.block_width, self.block_height)

                pygame.draw.rect(self.window, self.get_color(field[row][block]), rectangle)

        self.clock.tick(24)
        pygame.display.update()


Game()
