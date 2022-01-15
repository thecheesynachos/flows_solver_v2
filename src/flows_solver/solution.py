from typing import Tuple
import numpy as np

from flows_solver.constants import Coord


class FlowsSolution:

    def __init__(self, board_size: Tuple[int, int], colour_count: int):
        self.board_size = board_size
        self.colour_count = colour_count
        self.solution_board = np.zeros(shape=board_size, dtype=np.int32) - 1
        self.solution_edges = np.zeros(shape=(board_size[0] * board_size[1], 5), dtype=np.int32)
        self.num_edges = 0
        self.curr_path_length = np.zeros(shape=(colour_count,))

    def set_colour(self, coord: Coord, colour: int):
        self.solution_board[coord] = colour

    def clear_colour(self, coord: Coord):
        self.solution_board[coord] = -1

    def add_edge(self, colour: int, coord_from: Coord, coord_to: Coord):
        self.solution_edges[self.num_edges] = (colour, coord_from[0], coord_from[1], coord_to[0], coord_to[1])
        self.num_edges += 1
        self.curr_path_length[colour] += 1

    def pop_edge(self):
        self.num_edges -= 1
        c = self.solution_edges[self.num_edges][0]
        self.curr_path_length[c] -= 1

    def edge_iterator(self):
        for i in range(self.num_edges):
            yield self.solution_edges[i]
