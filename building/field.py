import numpy as np
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
        return " " + self.field.__repr__() # .replace("],", "],\n ")

field = Field(20, 20, 20, 20)
for frame in range(10):
    field.update()
    print(field)
