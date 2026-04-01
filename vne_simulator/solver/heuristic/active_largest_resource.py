from vne_simulator.base import Controller, Recorder, Counter, Solution, SolutionStepEnvironment
from vne_simulator.data import PhysicalNetwork, VirtualNetwork
from vne_simulator.solver import registry

from ..solver import Solver


@registry.register(
    solver_name='active_largest_resource',
    env_cls=SolutionStepEnvironment,
    solver_type='heuristic')
class ActiveLargestResourceSolver(Solver):
    """
    A conservative baseline used to sanity-check temporal substrate logic.

    The solver follows a simple policy:
    1. Work only on the active substrate exposed by the controller.
    2. For each virtual node, pick an active physical node with enough
       residual resources and the largest total residual node resource.
    3. For each virtual link, route on the active substrate with a
       feasible BFS shortest path.
    4. Accept iff every node and link can be embedded.
    """

    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, **kwargs) -> None:
        super(ActiveLargestResourceSolver, self).__init__(controller, recorder, counter, **kwargs)
        self.shortest_method = 'bfs_shortest'
        self.k_shortest = 1

    def solve(self, instance: dict) -> Solution:
        v_net, p_net = instance['v_net'], instance['p_net']
        solution = Solution(v_net)

        node_mapping_result = self.node_mapping(v_net, p_net, solution)
        if not node_mapping_result:
            solution['place_result'] = False
            solution['result'] = False
            return solution

        link_mapping_result = self.link_mapping(v_net, p_net, solution)
        if not link_mapping_result:
            solution['route_result'] = False
            solution['result'] = False
            return solution

        solution['result'] = True
        return solution

    def node_mapping(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution) -> bool:
        for v_node_id in list(v_net.nodes):
            p_node_id = self.select_physical_node(v_net, p_net, v_node_id, solution)
            if p_node_id is None:
                solution.update({'place_result': False, 'result': False})
                return False

            place_result, _ = self.controller.place(v_net, p_net, v_node_id, p_node_id, solution)
            if not place_result:
                solution.update({'place_result': False, 'result': False})
                return False

        return True

    def link_mapping(self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution) -> bool:
        sorted_v_links = list(v_net.links)
        link_mapping_result = self.controller.link_mapping(
            v_net,
            p_net,
            solution=solution,
            sorted_v_links=sorted_v_links,
            shortest_method=self.shortest_method,
            k=self.k_shortest,
            inplace=True,
        )
        return link_mapping_result

    def select_physical_node(
            self,
            v_net: VirtualNetwork,
            p_net: PhysicalNetwork,
            v_node_id: int,
            solution: Solution,
        ):
        used_p_nodes = list(solution['node_slots'].values())
        candidate_nodes = self.controller.find_candidate_nodes(
            v_net,
            p_net,
            v_node_id,
            filter=used_p_nodes,
            check_node_constraint=True,
            check_link_constraint=False,
        )
        if len(candidate_nodes) == 0:
            return None

        ranked_candidates = sorted(
            candidate_nodes,
            key=lambda p_node_id: (-self.score_physical_node(p_net, p_node_id), str(p_node_id)),
        )
        return ranked_candidates[0]

    def score_physical_node(self, p_net: PhysicalNetwork, p_node_id: int) -> float:
        return float(sum(
            p_net.nodes[p_node_id][n_attr.name]
            for n_attr in self.controller.node_resource_attrs
            if n_attr.name in p_net.nodes[p_node_id]
        ))
