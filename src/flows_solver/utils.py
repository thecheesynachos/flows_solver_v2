from typing import Tuple, Dict, List, Set

from flows_solver.constants import Coord


def manhattan_dist(p1: Coord, p2: Coord) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def generate_graph(board_size: Tuple[int, int]):
    adjacent_box = dict()
    for x in range(board_size[0]):
        for y in range(board_size[1]):
            adjacent_box[x, y] = [(x + dx, y) for dx in [-1, 1] if 0 <= x + dx < board_size[0]]
            adjacent_box[x, y].extend([(x, y + dy) for dy in [-1, 1] if 0 <= y + dy < board_size[1]])
    return adjacent_box


def shortest_path(graph: Dict[Coord, List[Coord]], source: Coord, target: Coord,
                  restricted_spaces: Set[Coord] = None) -> int:
    if source == target:
        return 0

    if restricted_spaces is None:
        restricted_spaces = set()

    covered_set = {source}
    level_set = {source}
    dist = 0
    while len(level_set) > 0:
        dist += 1
        new_level_set = set()
        for nd in level_set:
            nbrs = {nbr for nbr in graph[nd] if nbr not in covered_set and (nbr == target or
                                                                            nbr not in restricted_spaces)}
            new_level_set.update(nbrs)
            covered_set.update(nbrs)
        level_set = new_level_set
        if target in level_set:
            return dist
    raise ValueError('No path from source to target')
