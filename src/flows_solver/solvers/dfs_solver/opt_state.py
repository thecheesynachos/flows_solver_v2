from typing import List, Tuple, Dict
import numpy as np

from flows_solver.constants import Coord
from flows_solver.utils import shortest_path, manhattan_dist
from flows_solver.problem import FlowsProblem
from flows_solver.solution import FlowsSolution


class DFSOptimisationState:

    def __init__(self, board_size: Tuple[int, int], colour_pairs: List[Tuple[Coord, Coord]],
                 graph: Dict[Coord, List[Coord]], colour_order_mode: int = 0):
        self.board_size = board_size
        self.colour_pairs = colour_pairs
        self.graph = graph
        self.flows_solution = FlowsSolution(board_size=board_size, colour_count=len(colour_pairs))

        # coordinates that are fixed (i.e. coordinates that are a part of the colour pairs)
        self.fixed_coords = {nd for pair in self.colour_pairs for nd in pair}
        # should_avoid[i, j] = 1 if (i, j) in fixed_coords, else 0
        self.should_avoid = np.zeros(shape=board_size, dtype=np.bool8)
        for p in self.fixed_coords:
            self.should_avoid[p] = True

        # array to keep info about which colour pairs should be searched through first
        self.colour_order = list(range(len(colour_pairs)))
        if colour_order_mode == 0:
            pass
        elif colour_order_mode == 1 or colour_order_mode == -1:
            # sort the colours by how far the colour pairs are from each other
            # 1 = closest first, -1 = furthest first, 0 = don't sort
            self.colour_order.sort(key=lambda x: shortest_path(graph=self.graph,
                                                               source=self.colour_pairs[x][0],
                                                               target=self.colour_pairs[x][1],
                                                               restricted_spaces=self.fixed_coords),
                                   reverse=(colour_order_mode == -1))
        else:
            raise ValueError('Invalid colour_order_mode - must be -1, 0 or 1 only.')

        # number of colour pairs
        self.pairs_count = len(colour_pairs)

        # graph_ordered_by_distance[i] is adjacency array of the coordinates
        #       but with the coordinates in each list ordered by closest to the target of pair i first
        self.graph_ordered_by_distance: Dict[int, Dict[Coord, List[Coord]]] = dict()
        for c in range(len(colour_pairs)):
            self.graph_ordered_by_distance[c] = dict()
            for coord, nbrs in graph.items():
                if self.get_target(c) in nbrs:
                    nbrs_ordered = [self.get_target(c)]
                elif not self.should_avoid[coord] or coord == self.get_start(c):
                    nbrs_ordered = [nbr for nbr in nbrs if not self.should_avoid[nbr] or nbr == self.get_target(c)]
                    nbrs_ordered = sorted(nbrs_ordered, key=lambda x: manhattan_dist(x, self.get_target(c)))
                else:
                    nbrs_ordered = []
                self.graph_ordered_by_distance[c][coord] = nbrs_ordered

        # number of empty neighbours of each coordinate
        self.free_neighbours = np.empty(shape=board_size, dtype=np.int32)
        for coord, nbrs in graph.items():
            self.free_neighbours[coord] = len(nbrs)

        # empty board
        self.dummy_board = np.zeros(shape=board_size, dtype=np.int32) - 1
        # number of search states traversed
        self.search_states_encountered = 0
        # whether the current path is forced (i.e. previous coord has only one free space available)
        self.path_is_forced = False

    @staticmethod
    def from_problem(problem: FlowsProblem, colour_order_mode: int = 0) -> 'DFSOptimisationState':
        return DFSOptimisationState(board_size=problem.board_size,
                                    colour_pairs=problem.colour_pairs,
                                    graph=problem.graph,
                                    colour_order_mode=colour_order_mode)

    def increment_search_states_encountered(self):
        """ Increment number of states

        :return:
        """
        self.search_states_encountered += 1

    def get_search_states_encountered(self) -> int:
        """ get number of search states encountered

        :return: int
        """
        return self.search_states_encountered

    def get_adjacent_coords_of(self, coord: Coord, colour: int = 0) -> List[Coord]:
        """ get neighbours of certain coordinate

        :param coord: coordinate interested
        :param colour: sort the neighbours by closest distance to target of this colour pair
        :return:
        """
        return self.graph_ordered_by_distance[colour][coord]

    def get_start(self, colour: int) -> Coord:
        return self.colour_pairs[colour][0]

    def get_target(self, colour: int) -> Coord:
        return self.colour_pairs[colour][1]

    def verify_next_pos(self, next_coord: Coord, current_colour: int) -> bool:
        target = self.get_target(current_colour)
        if self.flows_solution.solution_board[next_coord] >= 0:
            return False
        elif not (not self.should_avoid[next_coord] or next_coord == target):
            return False
        else:
            return True

    def verify_no_parallel_path(self, next_coord: Coord, current_colour: int) -> bool:
        total_same = sum(1 for x in self.graph[next_coord] if self.flows_solution.solution_board[x] == current_colour)
        return not (total_same > 1)

    def _paint_one_side(self, start: Coord, tag: int):
        total_painted = 0
        if self.dummy_board[start] == 0:
            total_painted += 1
        self.dummy_board[start] = tag
        for nbr in self.graph[start]:
            if self.dummy_board[nbr] >= 0 and self.dummy_board[nbr] != tag:
                if self.should_avoid[nbr]:
                    if self.dummy_board[nbr] == 0:
                        total_painted += 1
                    self.dummy_board[nbr] = tag
                else:
                    total_painted += self._paint_one_side(nbr, tag)
        return total_painted

    def _verify_current_solution(self, colours_to_check: List[int], check_unreachable_space: bool = True) -> bool:

        np.copyto(self.dummy_board, -1 * (self.flows_solution.solution_board >= 0))
        required_paint = self.dummy_board.size + np.sum(self.dummy_board)

        tag = 1
        painted_num = 0

        for c in colours_to_check:
            pair = self.colour_pairs[c]
            if ((self.dummy_board[pair[0]] == 0) or (check_unreachable_space and required_paint > painted_num) or
                    (self.dummy_board[pair[0]] != self.dummy_board[pair[1]])):
                tag += 1
                painted_num += self._paint_one_side(start=pair[0], tag=tag)
            if self.dummy_board[pair[0]] != self.dummy_board[pair[1]]:
                return False

        if check_unreachable_space and (required_paint > painted_num):
            return False

        return True

    def verify_tight_space(self, coord: Coord):
        return not ((self.flows_solution.solution_board[coord] < 0) and (not self.should_avoid[coord]) and
                    (self.free_neighbours[coord] <= 1))

    def verify_tight_spaces_around(self, coord_centre: Coord):
        for nbr in self.graph[coord_centre]:
            if not self.verify_tight_space(nbr):
                return False
        return True

    def verify_all_tight_spaces(self):
        for coord in self.graph.keys():
            if not self.verify_tight_space(coord):
                return False
        return True

    """ Functions below here are to be used for following order of colours based on new ordering
    """

    def get_adjacent_coords_by_index(self, coord: Coord, colour_idx: int = 0):
        return self.get_adjacent_coords_of(coord, self.colour_order[colour_idx])

    def get_start_by_index(self, colour_index: int) -> Coord:
        return self.get_start(self.colour_order[colour_index])

    def get_target_by_index(self, colour_index: int) -> Coord:
        return self.get_target(self.colour_order[colour_index])

    def verify_next_pos_for_index(self, next_coord: Coord, colour_index: int):
        return self.verify_next_pos(next_coord=next_coord, current_colour=self.colour_order[colour_index])

    def verify_no_parallel_path_for_index(self, next_coord: Coord, colour_index: int):
        return self.verify_no_parallel_path(next_coord=next_coord, current_colour=self.colour_order[colour_index])

    def verify_current_solution_from_index(self, colour_index: int, check_unreachable_space: bool = True):
        return self._verify_current_solution(self.colour_order[colour_index:], check_unreachable_space)

    """ Solutions manipulating functions
    """

    def set_colour_by_index(self, coord: Coord, colour_index: int):
        self.flows_solution.set_colour(coord, self.colour_order[colour_index])
        for nbr in self.graph[coord]:
            self.free_neighbours[nbr] -= 1

    def clear_colour(self, coord: Coord):
        self.flows_solution.clear_colour(coord)
        for nbr in self.graph[coord]:
            self.free_neighbours[nbr] += 1

    def add_edge_by_index(self, colour_index: int, coord_from: Coord, coord_to: Coord):
        self.flows_solution.add_edge(colour=self.colour_order[colour_index],
                                     coord_from=coord_from,
                                     coord_to=coord_to)

    def pop_edge(self):
        self.flows_solution.pop_edge()
