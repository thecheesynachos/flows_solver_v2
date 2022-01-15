from typing import Tuple, Any, Union

from flows_solver.problem import FlowsProblem
from flows_solver.solution import FlowsSolution


class FlowsSolver:

    def solve(self, problem: FlowsProblem) -> FlowsSolution:
        raise NotImplementedError
