from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from flows_solver.constants import Coord
from flows_solver.utils import generate_graph, shortest_path
from flows_solver.solution import FlowsSolution


class FlowsProblem:

    def __init__(self, board_size: Tuple[int, int], colour_pairs: List[Tuple[Coord, Coord]],
                 graph: Dict[Coord, List[Coord]] = None):
        self.board_size = board_size
        self.colour_pairs = colour_pairs
        self.colours_count = len(colour_pairs)
        self.graph = graph if graph is not None else generate_graph(self.board_size)
        self.fixed_coords = {nd for pair in self.colour_pairs for nd in pair}

    @staticmethod
    def from_board_array(board_array: np.ndarray):
        board_size = board_array.shape
        assert len(board_size) == 2  # 2 dimensional array
        pairs_dict = dict()
        pairs_count = 0
        for i in range(board_size[0]):
            for j in range(board_size[1]):
                c = board_array[i, j]
                if c != 0:
                    if c in pairs_dict.keys():
                        pairs_dict[c] = ((i, j), (pairs_dict[c]))
                    else:
                        pairs_dict[c] = (i, j)
                        pairs_count += 1
        pairs = [pairs_dict[c] for c in range(1, pairs_count + 1)]
        return FlowsProblem(board_size=board_size, colour_pairs=pairs)

    def plot_game(self, solution: FlowsSolution = None,
                  current_interested_point: Coord = None, additional_mark_points: List[Coord] = None):
        fig = plt.figure(figsize=(self.board_size[0] + 1, self.board_size[1] + 1))
        ax = fig.add_subplot()
        ax.set_xlim(-0.5, self.board_size[0] - 0.5)
        ax.set_ylim(-0.5, self.board_size[1] - 0.5)

        for i, ((x1, y1), (x2, y2)) in enumerate(self.colour_pairs):
            ax.text(x1, y1, chr(65 + i), verticalalignment='center', horizontalalignment='center',
                    fontsize=25, zorder=40)
            ax.text(x2, y2, chr(65 + i), verticalalignment='center', horizontalalignment='center',
                    fontsize=25, zorder=40)
            ax.plot([x1, x2], [y1, y2], 'o', color=f'C{i}', markersize=40, zorder=30)

        if solution is not None:
            for (i, x1, y1, x2, y2) in solution.edge_iterator():
                ax.plot([x1, x2], [y1, y2], color=f'C{i}', linewidth=15, zorder=20)

        if current_interested_point is not None:
            ax.plot([current_interested_point[0]], [current_interested_point[1]],
                    'o', color='black', markersize=50, zorder=10, alpha=0.3)

        if additional_mark_points is not None:
            ax.plot([p[0] for p in additional_mark_points], [p[1] for p in additional_mark_points],
                    'x', color='black', markersize=30, zorder=50, alpha=0.7)

        plt.close(fig)
        return fig
