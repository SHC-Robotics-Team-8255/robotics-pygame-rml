import numpy as np
import random


class Field:

    def __init__(self, row_blocks, col_blocks, block_width, block_height):
        self.row_width = row_blocks
        self.col_height = col_blocks

        self.block_width = block_width
        self.block_height = block_height

        self.total_width = None  # fill these two in. Hint: multiply
        self.total_height = None

        self.field = np.zeros((col_blocks, row_blocks), int)  # lets make an empty 2d array to represent our squares

        self.gate_width = 5
        self.padding = 2

        self.layers_per_gate = 2  # gate thickness

        self.next_gate = True
        # in order to both produce a gate now and later we enter into the production phase right away
        self.layers_left = self.layers_per_gate

        self.gate = self.generate_gate()  # make our first gate!

    def generate_gate(self):
        gate = np.ones(self.row_width)

        # we are looking for the acceptable x-coordinate we can start to make a gate with the length of 5
        start = None
        end = None
        gate_left_start = random.randrange(start, end)

        for i in range(gate_left_start, gate_left_start + self.gate_width):
            gate[i] = 0  # here, we make the gate

        return gate

    def update(self):
        if self.field[19][0] == 1 and not self.next_gate:  # check if we ran out of gates
            self.next_gate = True
            self.gate = self.generate_gate()  # make a new gate if so

        self.field = np.delete(self.field, 19, 0)  # delete the last row, its off the screen now
        if self.next_gate:
            self.field = np.insert(self.field, 0, self.gate, 0)  # push a gate to the screen
            self.layers_left -= 1
            if self.layers_left == 0:  # finished adding our gate, lets reset
                self.next_gate = False
                self.layers_left = self.layers_per_gate
        else:
            self.field = np.insert(self.field, 0, np.zeros(self.row_width, int), 0)  # push a normal empty row to the screen

    def __repr__(self):  # custom print function
        return " " + self.field.__repr__()  # .replace("],", "],\n ")


field = Field(20, 20, 20, 20)
for frame in range(10):
    field.update()
    print(field)
