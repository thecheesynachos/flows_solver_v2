from typing import Union, Tuple, Any
import matplotlib.pyplot as plt
import os

from flows_solver.constants import Coord
from flows_solver.problem import FlowsProblem
from flows_solver.solvers.dfs_solver.opt_state import DFSOptimisationState
from flows_solver.solution import FlowsSolution
from flows_solver.solvers.abstract_solver import FlowsSolver


class DFSSolver(FlowsSolver):

    def __init__(self, do_closest_pairs_first: bool = True,
                 do_furthest_pairs_first: bool = False,
                 sort_adjacency_list_by_dist: bool = True,
                 verify_parallel_paths: bool = True,
                 verify_tight_spaces: bool = True,
                 verify_global_tight_spaces: bool = True,
                 verify_solution_after_path_complete: bool = True,
                 verify_solution_after_bottleneck: bool = True,
                 save_progress_images_mode: int = 0,
                 save_progress_images_directory: str = None):
        """

        :param do_closest_pairs_first: search for path for closest colour pairs first
        :param do_furthest_pairs_first: search for path for furthest colour pairs first
        :param sort_adjacency_list_by_dist: in searching stage, perform search on stages where
                the next step is closer to target coordinate first
        :param verify_parallel_paths: whether to check for parallel (winding) paths or not
        :param verify_tight_spaces: whether to check for tight spaces (empty spaces with <= 1 nbrs)
                after every added path step
        :param verify_global_tight_spaces: whether to check for tight spaces (empty spaces with <= 1 nbrs)
                after every solution completion
        :param verify_solution_after_path_complete: whether to check for unreachable empty spaces
                after each connected path
        :param verify_solution_after_bottleneck: whether to check for unreachable empty spaces
                after each pathline bottleneck reached
        :param save_progress_images_directory: where to save the image of each solution progression
        """
        assert not (do_closest_pairs_first and do_furthest_pairs_first)
        assert save_progress_images_mode in [0, 1, 2]
        assert (save_progress_images_mode == 0) or (save_progress_images_directory is not None)

        self.do_closest_pairs_first = do_closest_pairs_first
        self.do_furthest_pairs_first = do_furthest_pairs_first
        self.sort_adjacency_list_by_dist = sort_adjacency_list_by_dist
        self.verify_parallel_paths = verify_parallel_paths
        self.verify_tight_spaces = verify_tight_spaces
        self.verify_global_tight_spaces = verify_global_tight_spaces
        self.verify_solution_after_path_complete = verify_solution_after_path_complete
        self.verify_solution_after_bottleneck = verify_solution_after_bottleneck
        self.save_progress_images_mode = save_progress_images_mode
        self.save_progress_images_directory = save_progress_images_directory

        self.opt_state: DFSOptimisationState = None
        self._pics_saved = 0

    def _solver_helper(self, problem: FlowsProblem, current_colour: int,
                       current_coord: Coord, prev_coord: Coord = None):

        self.opt_state.set_colour_by_index(coord=current_coord, colour_index=current_colour)
        target = self.opt_state.get_target_by_index(colour_index=current_colour)

        # for saving the current game state at the moment
        if self.save_progress_images_mode == 1 or self.save_progress_images_mode == 2:
            problem.plot_game(
                solution=self.opt_state.flows_solution,
                current_interested_point=current_coord
            ).savefig(os.path.join(
                self.save_progress_images_directory, f'state_{self._pics_saved:06d}'))
            self._pics_saved += 1

        # increase number of states checked
        self.opt_state.increment_search_states_encountered()

        current_path_was_forced = self.opt_state.path_is_forced
        still_ok_flag = True

        # fail condition here if
        # 1. user set verify_tight_spaces = True
        # 2. prev_coord is not empty (this isn't the start of new path)
        # 3. some neighbour of prev_coord is blank and has one or fewer free neighbour (tight spaces)
        if (self.verify_tight_spaces and (prev_coord is not None) and
                not self.opt_state.verify_tight_spaces_around(prev_coord)):
            still_ok_flag = False

        if still_ok_flag:
            if current_coord == target:
                if current_colour < problem.colours_count - 1:
                    current_colour += 1
                    if ((not self.verify_global_tight_spaces or self.opt_state.verify_all_tight_spaces()) and
                            (not self.verify_solution_after_path_complete or
                             self.opt_state.verify_current_solution_from_index(colour_index=current_colour))):
                        # enter condition here (i.e. not failed yet) if
                        # 1. user set verify_global_tight_spaces = True
                        # 2. there are no empty spaces with one or fewer free neighbours (tight spaces)
                        # 3. all colour pairs are still reachable from each other (check current solution)
                        # 4. no empty spaces unreachable by a colour pair (also check current solution)
                        self.opt_state.path_is_forced = False
                        status = self._solver_helper(problem=problem,
                                                     current_colour=current_colour,
                                                     current_coord=self.opt_state.get_start_by_index(current_colour))
                        if status:
                            # if return status is true, found solution
                            return True

                else:
                    # solution still works and no more colours left to draw paths between
                    # so the solution is now found
                    return True

            else:

                if self.sort_adjacency_list_by_dist:
                    # in the case that the neighbours should be checked in distance order
                    nbrs_iterator = self.opt_state.get_adjacent_coords_by_index(coord=current_coord,
                                                                                colour_idx=current_colour)
                else:
                    # in the case that the neighbours can be checked however
                    nbrs_iterator = self.opt_state.graph[current_coord]

                # neighbours will be checked if
                # 1. square is empty (verify next pos)
                # 2. if verify_parallel_paths = True, also check that
                #       there are no extra winding of path (verify parallel path)
                nbrs_iterator = [next_coord for next_coord in nbrs_iterator
                                 if (self.opt_state.verify_next_pos_for_index(next_coord=next_coord,
                                                                              colour_index=current_colour) and
                                     (not self.verify_parallel_paths or
                                      self.opt_state.verify_no_parallel_path_for_index(
                                          next_coord=next_coord, colour_index=current_colour)))
                                 ]

                self.opt_state.path_is_forced = (len(nbrs_iterator) <= 1)

                if not (self.verify_solution_after_bottleneck and len(nbrs_iterator) <= 1 and
                        current_colour < self.opt_state.pairs_count - 1 and
                        not current_path_was_forced and
                        not self.opt_state.verify_current_solution_from_index(colour_index=current_colour + 1,
                                                                              check_unreachable_space=False)):
                    # enter condition here if
                    # 1. if verify_solution_after_bottleneck = True, and there are only one neighbour, then check that
                    #       all colours are still reachable from each other, and all empty squares are reachable
                    #       by a colour pair (verify current solution)
                    for i, next_coord in enumerate(nbrs_iterator):

                        # for drawing diagram of current state and available actions
                        if self.save_progress_images_mode == 2:
                            problem.plot_game(
                                solution=self.opt_state.flows_solution,
                                current_interested_point=current_coord,
                                additional_mark_points=nbrs_iterator[i:]
                            ).savefig(os.path.join(
                                self.save_progress_images_directory, f'state_{self._pics_saved:06d}'))
                            self._pics_saved += 1

                        # add edge from current coord to a neighbour
                        self.opt_state.add_edge_by_index(colour_index=current_colour,
                                                         coord_from=current_coord,
                                                         coord_to=next_coord)
                        # recursively check solution further
                        status = self._solver_helper(problem=problem,
                                                     current_colour=current_colour,
                                                     current_coord=next_coord,
                                                     prev_coord=current_coord)
                        if status:
                            # found solution
                            return True
                        else:
                            # if breaks, then pop edge and retry next nbr
                            self.opt_state.pop_edge()

        # restore path_is_forced variable
        self.opt_state.path_is_forced = current_path_was_forced
        # clear the colour of current coord
        self.opt_state.clear_colour(current_coord)
        # declare failure in this subtree
        return False

    def solve(self, problem: FlowsProblem) -> FlowsSolution:

        if self.do_closest_pairs_first:
            # do colour pairs by closest pairs first
            colour_order_mode = 1
        elif self.do_furthest_pairs_first:
            # do colour pairs by furthest pairs first
            colour_order_mode = -1
        else:
            # do colour pairs by whatever the input order is
            colour_order_mode = 0

        if self.save_progress_images_mode != 0:
            os.makedirs(self.save_progress_images_directory, exist_ok=True)

        self.opt_state = DFSOptimisationState.from_problem(problem=problem,
                                                           colour_order_mode=colour_order_mode)
        status = self._solver_helper(problem=problem,
                                     current_colour=0,
                                     current_coord=self.opt_state.get_start_by_index(0))
        if status:
            return self.opt_state.flows_solution
        else:
            raise ValueError('Problem infeasible.')
