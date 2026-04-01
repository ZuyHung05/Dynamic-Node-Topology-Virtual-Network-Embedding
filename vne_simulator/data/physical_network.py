from functools import cached_property
import os
import copy
import random
import numpy as np
import networkx as nx

from vne_simulator.utils import read_setting, write_setting
from .network import Network
from .attribute import Attribute, NodeInfoAttribute, LinkInfoAttribute


class PhysicalNetwork(Network):
    """
    PhysicalNetwork class is a subclass of Network class. It represents a physical network.

    Attributes:
        degree_benchmark (dict): The degree benchmark for the network.
        node_attr_benchmarks (dict): The node attribute benchmarks for the network.
        link_attr_benchmarks (dict): The link attribute benchmarks for the network.

    Methods:
        from_topology_zoo_setting(topology_zoo_setting: Dict[str, Any], seed: Optional[int] = None) -> PhysicalNetwork:
            Returns a PhysicalNetwork object generated from the Topology Zoo data, with optional seed for reproducibility.

        save_dataset(self, dataset_dir: str) -> None:
            Saves the dataset as a .gml file in the specified directory.

        load_dataset(dataset_dir: str) -> PhysicalNetwork:
            Loads the dataset from the specified directory as a PhysicalNetwork object, 
            and calculates the benchmarks for normalization.
    """
    def __init__(self, incoming_graph_data: nx.Graph = None, node_attrs_setting: list = [], link_attrs_setting: list = [], **kwargs) -> None:
        """
        Initialize a PhysicalNetwork object.

        Args:
            incoming_graph_data (nx.Graph): An existing graph object (optional).
            node_attrs_setting (list): Node attribute settings (default []).
            link_attrs_setting (list): Link attribute settings (default []).
            **kwargs: Additional keyword arguments.
        """
        super(PhysicalNetwork, self).__init__(incoming_graph_data, node_attrs_setting, link_attrs_setting, **kwargs)
        # Keep the graph as the candidate substrate topology and store
        # time-varying substrate state in auxiliary containers.
        self.static_nodes = set(kwargs.get('static_nodes', []))
        self.dynamic_node_candidates = set(kwargs.get('dynamic_node_candidates', []))
        self.static_links = {tuple(link) for link in kwargs.get('static_links', [])}
        self.dynamic_link_candidates = {tuple(link) for link in kwargs.get('dynamic_link_candidates', [])}
        self.time_slots = list(kwargs.get('time_slots', []))
        self.node_activation = copy.deepcopy(kwargs.get('node_activation', {}))
        self.link_activation = copy.deepcopy(kwargs.get('link_activation', {}))
        self.active_time_slot = kwargs.get('active_time_slot')
        self.active_event_id = kwargs.get('active_event_id')

    @staticmethod
    def _canonical_link(link):
        """Normalize an undirected link representation for comparisons."""
        u, v = tuple(link)
        return tuple(sorted((u, v)))

    def _resolve_candidate_partitions(self, all_nodes, all_links, **kwargs):
        """
        Split the candidate substrate graph into static and dynamic candidates.

        If no partition is provided, keep the old behavior by treating all
        nodes and links as static.
        """
        static_nodes = kwargs.get('static_nodes', kwargs.get('static_node_ids'))
        dynamic_nodes = kwargs.get('dynamic_node_candidates', kwargs.get('dynamic_node_ids'))
        force_all_links_static = kwargs.get('all_links_static', self.graph.get('all_links_static', False))

        if static_nodes is None and dynamic_nodes is None:
            static_nodes = set(all_nodes)
            dynamic_nodes = set()
        else:
            static_nodes = set(all_nodes if static_nodes is None else static_nodes)
            dynamic_nodes = set() if dynamic_nodes is None else set(dynamic_nodes)
            if kwargs.get('static_nodes', kwargs.get('static_node_ids')) is None:
                static_nodes = set(all_nodes) - dynamic_nodes
            if kwargs.get('dynamic_node_candidates', kwargs.get('dynamic_node_ids')) is None:
                dynamic_nodes = set(all_nodes) - static_nodes

        if not static_nodes.issubset(all_nodes) or not dynamic_nodes.issubset(all_nodes):
            raise ValueError('Node partitions must be subsets of the generated topology.')
        if static_nodes & dynamic_nodes:
            raise ValueError('Static and dynamic node partitions must be disjoint.')

        canonical_to_link = {self._canonical_link(link): tuple(link) for link in all_links}
        static_links = kwargs.get('static_links')
        dynamic_links = kwargs.get('dynamic_link_candidates', kwargs.get('dynamic_links'))

        if force_all_links_static:
            static_links = set(all_links)
            dynamic_links = set()
        elif static_links is None and dynamic_links is None:
            static_links = set(all_links)
            dynamic_links = set()
        else:
            static_links = set(all_links if static_links is None else {
                canonical_to_link[self._canonical_link(link)] for link in static_links
            })
            dynamic_links = set() if dynamic_links is None else {
                canonical_to_link[self._canonical_link(link)] for link in dynamic_links
            }
            if kwargs.get('static_links') is None:
                static_links = set(all_links) - dynamic_links
            if kwargs.get('dynamic_link_candidates', kwargs.get('dynamic_links')) is None:
                dynamic_links = set(all_links) - static_links

        if not static_links.issubset(all_links) or not dynamic_links.issubset(all_links):
            raise ValueError('Link partitions must be subsets of the generated topology.')
        if static_links & dynamic_links:
            raise ValueError('Static and dynamic link partitions must be disjoint.')

        return static_nodes, dynamic_nodes, static_links, dynamic_links

    def _annotate_candidate_roles(self):
        """Mark each node and link with its candidate substrate role."""
        node_roles = {
            node_id: ('static' if node_id in self.static_nodes else 'dynamic_candidate')
            for node_id in self.nodes
        }
        nx.set_node_attributes(self, node_roles, 'candidate_role')

        link_roles = {
            tuple(link): ('static' if tuple(link) in self.static_links else 'dynamic_candidate')
            for link in self.links
        }
        nx.set_edge_attributes(self, link_roles, 'candidate_role')

    def _ensure_node_resource_attribute(self, name, fallback_value=1.0):
        """Ensure a node-level candidate capacity exists on every substrate node."""
        if name not in self.node_attrs:
            self.node_attrs[name] = Attribute.from_dict({
                'name': name,
                'owner': 'node',
                'type': 'resource',
                'generative': False,
            })
        attr_data = nx.get_node_attributes(self, name)
        if len(attr_data) == self.number_of_nodes():
            return
        default_value = self.graph.get(f'default_{name}', fallback_value)
        completed_data = {
            node_id: attr_data.get(node_id, default_value)
            for node_id in self.nodes
        }
        nx.set_node_attributes(self, completed_data, name)

    def _ensure_node_info_attribute(self, name, nodes, fallback_value=0.0):
        """Attach an operational parameter to selected substrate nodes."""
        if name not in self.node_attrs:
            self.node_attrs[name] = NodeInfoAttribute(name)
        attr_data = nx.get_node_attributes(self, name)
        default_value = self.graph.get(f'default_{name}', fallback_value)
        for node_id in nodes:
            if node_id not in attr_data:
                attr_data[node_id] = default_value
        nx.set_node_attributes(self, attr_data, name)

    def _ensure_link_info_attribute(self, name, links, fallback_value=0.0):
        """Attach an operational parameter to selected substrate links."""
        if name not in self.link_attrs:
            self.link_attrs[name] = LinkInfoAttribute(name)
        attr_data = nx.get_edge_attributes(self, name)
        default_value = self.graph.get(f'default_{name}', fallback_value)
        for link in links:
            if link not in attr_data:
                attr_data[link] = default_value
        nx.set_edge_attributes(self, attr_data, name)

    def _initialize_dynamic_node_parameters(self):
        """Ensure dynamic substrate nodes carry memory and energy parameters."""
        self._ensure_node_resource_attribute('memory', fallback_value=1.0)
        dynamic_nodes = self.dynamic_node_candidates or set()
        self._ensure_node_info_attribute('base_energy', dynamic_nodes, fallback_value=0.0)
        self._ensure_node_info_attribute('cpu_energy_coeff', dynamic_nodes, fallback_value=0.0)
        self._ensure_node_info_attribute('memory_energy_coeff', dynamic_nodes, fallback_value=0.0)
        self._ensure_node_info_attribute('power_on_energy', dynamic_nodes, fallback_value=0.0)
        self._ensure_node_info_attribute('power_off_energy', dynamic_nodes, fallback_value=0.0)

    def _initialize_dynamic_link_parameters(self):
        """Ensure dynamic substrate links carry topology-energy parameters."""
        dynamic_links = self.dynamic_link_candidates or set()
        self._ensure_link_info_attribute('link_active_energy', dynamic_links, fallback_value=0.0)
        self._ensure_link_info_attribute('link_up_energy', dynamic_links, fallback_value=0.0)
        self._ensure_link_info_attribute('link_down_energy', dynamic_links, fallback_value=0.0)

    def _initialize_activation_states(self):
        """Initialize temporal activation state for dynamic nodes and links."""
        if not self.time_slots:
            return

        completed_node_activation = {}
        for t in self.time_slots:
            time_node_state = copy.deepcopy(self.node_activation.get(t, {}))
            for node_id in self.dynamic_node_candidates:
                time_node_state.setdefault(node_id, 1)
            completed_node_activation[t] = time_node_state
        self.node_activation = completed_node_activation

        completed_link_activation = {}
        for t in self.time_slots:
            time_link_state = {}
            raw_state = copy.deepcopy(self.link_activation.get(t, {}))
            canonical_state = {
                self._canonical_link(link): value
                for link, value in raw_state.items()
            }
            for link in self.dynamic_link_candidates:
                time_link_state[link] = canonical_state.get(self._canonical_link(link), 1)
            completed_link_activation[t] = time_link_state
        self.link_activation = completed_link_activation

    def _temporal_metadata_keys(self):
        """Return graph-level temporal-environment keys to persist."""
        return [
            'all_links_static',
            'node_activation_mode',
            'dynamic_node_active_prob',
            'dynamic_node_initial_active_prob',
            'dynamic_node_on_to_off_prob',
            'dynamic_node_off_to_on_prob',
            'resample_node_activation_on_event',
        ]

    def _get_activation_rng(self, time_slot=None, event_id=None, seed=None):
        """Build a deterministic RNG for environment-driven node activation."""
        base_seed = seed if seed is not None else self.graph.get('seed', None)
        if base_seed is None:
            return np.random.default_rng()
        token = event_id if event_id is not None else self.resolve_time_slot(time_slot)
        token_str = f'{base_seed}:{token}'
        hashed_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(token_str)) % (2 ** 32)
        return np.random.default_rng(hashed_seed)

    def sample_node_activation(self, time_slot=None, event_id=None, seed=None, force=False):
        """Sample dynamic-node on/off states for the current event/time slot."""
        resolved_time = self.resolve_time_slot(time_slot)
        if resolved_time is None or not self.dynamic_node_candidates:
            return copy.deepcopy(self.node_activation.get(resolved_time, {}))

        mode = self.graph.get('node_activation_mode', 'precomputed')
        should_resample = force or self.graph.get('resample_node_activation_on_event', True)
        if mode in ['precomputed', None] and not should_resample:
            return copy.deepcopy(self.node_activation.get(resolved_time, {}))
        if mode in ['precomputed', None] and resolved_time in self.node_activation and self.node_activation[resolved_time]:
            return copy.deepcopy(self.node_activation[resolved_time])

        rng = self._get_activation_rng(resolved_time, event_id=event_id, seed=seed)
        previous_state = {}
        sorted_times = sorted(self.node_activation.keys(), key=lambda t: float(t))
        for known_time in sorted_times:
            try:
                if float(known_time) < float(resolved_time):
                    previous_state = self.node_activation[known_time]
            except (TypeError, ValueError):
                if known_time != resolved_time:
                    previous_state = self.node_activation[known_time]

        sampled_state = {}
        if mode == 'markov':
            initial_active_prob = self.graph.get(
                'dynamic_node_initial_active_prob',
                self.graph.get('dynamic_node_active_prob', 1.0),
            )
            on_to_off_prob = self.graph.get('dynamic_node_on_to_off_prob', 0.0)
            off_to_on_prob = self.graph.get('dynamic_node_off_to_on_prob', 1.0)
            for node_id in self.dynamic_node_candidates:
                if previous_state:
                    was_active = bool(previous_state.get(node_id, 0))
                    if was_active:
                        sampled_state[node_id] = int(rng.random() >= on_to_off_prob)
                    else:
                        sampled_state[node_id] = int(rng.random() < off_to_on_prob)
                else:
                    sampled_state[node_id] = int(rng.random() < initial_active_prob)
        else:
            active_prob = self.graph.get('dynamic_node_active_prob', 1.0)
            for node_id in self.dynamic_node_candidates:
                sampled_state[node_id] = int(rng.random() < active_prob)

        self.node_activation[resolved_time] = sampled_state
        return copy.deepcopy(sampled_state)

    def update_event_state(self, time_slot=None, event_id=None, seed=None):
        """Update the substrate state for the current event."""
        resolved_time = self.resolve_time_slot(time_slot)
        self.active_time_slot = resolved_time
        self.active_event_id = event_id
        mode = self.graph.get('node_activation_mode', 'precomputed')
        if mode in ['random', 'random_independent', 'markov']:
            self.sample_node_activation(resolved_time, event_id=event_id, seed=seed, force=True)
        elif resolved_time is not None and resolved_time not in self.node_activation and self.dynamic_node_candidates:
            self.node_activation[resolved_time] = {node_id: 1 for node_id in self.dynamic_node_candidates}
        return resolved_time

    def resolve_time_slot(self, time_slot=None):
        """Resolve a runtime time slot against the configured planning horizon."""
        candidate_time = self.active_time_slot if time_slot is None else time_slot
        if not self.time_slots:
            return candidate_time
        if candidate_time in self.time_slots:
            return candidate_time
        try:
            if int(candidate_time) in self.time_slots:
                return int(candidate_time)
        except (TypeError, ValueError):
            pass
        try:
            if float(candidate_time) in self.time_slots:
                return float(candidate_time)
        except (TypeError, ValueError):
            pass
        try:
            return min(self.time_slots, key=lambda t: abs(float(t) - float(candidate_time)))
        except (TypeError, ValueError):
            return self.time_slots[0]

    def is_node_active(self, node_id, time_slot=None):
        """Check whether a substrate node is active at the given time slot."""
        if node_id in self.static_nodes:
            return True
        if node_id not in self.dynamic_node_candidates:
            return node_id in self.nodes
        resolved_time = self.resolve_time_slot(time_slot)
        if resolved_time is None:
            return True
        return bool(self.node_activation.get(resolved_time, {}).get(node_id, 0))

    def is_link_active(self, link, time_slot=None):
        """Check whether a substrate link is active at the given time slot."""
        link = tuple(link)
        canonical_link = self._canonical_link(link)
        canonical_static = {self._canonical_link(item) for item in self.static_links}
        if canonical_link in canonical_static:
            return self.is_node_active(link[0], time_slot) and self.is_node_active(link[1], time_slot)
        canonical_dynamic = {self._canonical_link(item): tuple(item) for item in self.dynamic_link_candidates}
        if canonical_link not in canonical_dynamic:
            return (link in self.links or (link[1], link[0]) in self.links) and \
                self.is_node_active(link[0], time_slot) and self.is_node_active(link[1], time_slot)
        resolved_time = self.resolve_time_slot(time_slot)
        if resolved_time is None:
            return True
        active_value = self.link_activation.get(resolved_time, {}).get(canonical_dynamic[canonical_link], 0)
        return bool(active_value) and self.is_node_active(link[0], time_slot) and self.is_node_active(link[1], time_slot)

    def get_active_nodes(self, time_slot=None):
        """Return the active substrate node ids at the given time slot."""
        resolved_time = self.resolve_time_slot(time_slot)
        return [node_id for node_id in self.nodes if self.is_node_active(node_id, resolved_time)]

    def get_active_links(self, time_slot=None):
        """Return the active substrate links at the given time slot."""
        resolved_time = self.resolve_time_slot(time_slot)
        return [tuple(link) for link in self.links if self.is_link_active(link, resolved_time)]

    def get_active_subgraph(self, time_slot=None):
        """Return a view of the active substrate graph at the given time slot."""
        resolved_time = self.resolve_time_slot(time_slot)

        def filter_node(node_id):
            return self.is_node_active(node_id, resolved_time)

        def filter_edge(node_a, node_b):
            return self.is_link_active((node_a, node_b), resolved_time)

        return self.get_subgraph_view(filter_node=filter_node, filter_edge=filter_edge)

    def _apply_candidate_partition_metadata(self, partition_setting=None):
        """Apply static/dynamic candidate partitions to the current graph."""
        partition_setting = partition_setting or {}
        all_nodes = set(self.nodes)
        all_links = {tuple(link) for link in self.links}
        static_nodes, dynamic_nodes, static_links, dynamic_links = self._resolve_candidate_partitions(
            all_nodes,
            all_links,
            **partition_setting,
        )
        self.static_nodes = static_nodes
        self.dynamic_node_candidates = dynamic_nodes
        self.static_links = static_links
        self.dynamic_link_candidates = dynamic_links
        self._annotate_candidate_roles()

    def _apply_temporal_metadata(self, temporal_setting=None):
        """Apply temporal substrate state and defaults to the current graph."""
        temporal_setting = copy.deepcopy(temporal_setting or {})
        self.time_slots = list(temporal_setting.get('time_slots', self.time_slots))
        self.node_activation = copy.deepcopy(temporal_setting.get('node_activation', self.node_activation))
        self.link_activation = copy.deepcopy(temporal_setting.get('link_activation', self.link_activation))
        self.graph.update(temporal_setting.get('defaults', {}))
        for key in self._temporal_metadata_keys():
            if key in temporal_setting:
                self.graph[key] = temporal_setting[key]
        if self.graph.get('all_links_static', False):
            self.static_links = {tuple(link) for link in self.links}
            self.dynamic_link_candidates = set()
            self._annotate_candidate_roles()

    def _load_graph_data(self, graph):
        """Copy graph, node, and adjacency structures from a loaded graph."""
        if 'node_attrs_setting' in graph.__dict__['graph']:
            graph.__dict__['graph'].pop('node_attrs_setting')
        if 'link_attrs_setting' in graph.__dict__['graph']:
            graph.__dict__['graph'].pop('link_attrs_setting')
        self.__dict__['graph'].update(graph.__dict__['graph'])
        self.__dict__['_node'] = graph.__dict__['_node']
        self.__dict__['_adj'] = graph.__dict__['_adj']

        if self.number_of_nodes() > 0:
            n_attr_names = self.nodes[list(self.nodes)[0]].keys()
            for n_attr_name in n_attr_names:
                if n_attr_name not in self.node_attrs.keys():
                    net_attr = NodeInfoAttribute(n_attr_name)
                    self.node_attrs[n_attr_name] = net_attr
        if self.number_of_edges() > 0:
            l_attr_names = self.links[list(self.links)[0]].keys()
            for l_attr_name in l_attr_names:
                if l_attr_name not in self.link_attrs.keys():
                    net_attr = LinkInfoAttribute(l_attr_name)
                    self.link_attrs[l_attr_name] = net_attr

    @staticmethod
    def _extract_temporal_setting(setting):
        """Extract temporal substrate metadata from a config dictionary."""
        temporal_setting = copy.deepcopy(setting.pop('temporal', {}))

        if 'time_slots' in setting and 'time_slots' not in temporal_setting:
            temporal_setting['time_slots'] = copy.deepcopy(setting.pop('time_slots'))
        if 'node_activation' in setting and 'node_activation' not in temporal_setting:
            temporal_setting['node_activation'] = copy.deepcopy(setting.pop('node_activation'))
        if 'link_activation' in setting and 'link_activation' not in temporal_setting:
            temporal_setting['link_activation'] = copy.deepcopy(setting.pop('link_activation'))

        default_keys = [key for key in list(setting.keys()) if key.startswith('default_')]
        if default_keys:
            defaults = temporal_setting.setdefault('defaults', {})
            for key in default_keys:
                defaults[key] = setting.pop(key)

        metadata_keys = [
            'all_links_static',
            'node_activation_mode',
            'dynamic_node_active_prob',
            'dynamic_node_initial_active_prob',
            'dynamic_node_on_to_off_prob',
            'dynamic_node_off_to_on_prob',
            'resample_node_activation_on_event',
        ]
        for key in metadata_keys:
            if key in setting and key not in temporal_setting:
                temporal_setting[key] = copy.deepcopy(setting.pop(key))

        return temporal_setting

    def _serialize_temporal_metadata(self):
        """Serialize temporal substrate metadata into a JSON-friendly structure."""
        return {
            'static_nodes': sorted(self.static_nodes),
            'dynamic_node_candidates': sorted(self.dynamic_node_candidates),
            'static_links': [list(link) for link in sorted(self.static_links)],
            'dynamic_link_candidates': [list(link) for link in sorted(self.dynamic_link_candidates)],
            'time_slots': list(self.time_slots),
            'node_activation': {
                str(t): [[node_id, value] for node_id, value in sorted(time_state.items())]
                for t, time_state in self.node_activation.items()
            },
            'link_activation': {
                str(t): [[list(link), value] for link, value in sorted(time_state.items())]
                for t, time_state in self.link_activation.items()
            },
            'defaults': {
                key: value
                for key, value in self.graph.items()
                if isinstance(key, str) and key.startswith('default_')
            },
            **{
                key: self.graph[key]
                for key in self._temporal_metadata_keys()
                if key in self.graph
            },
        }

    @staticmethod
    def _coerce_node_id(node_id, all_nodes):
        """Convert serialized node ids back to the graph node type when possible."""
        if node_id in all_nodes:
            return node_id
        try:
            coerced = int(node_id)
            if coerced in all_nodes:
                return coerced
        except (TypeError, ValueError):
            pass
        return node_id

    @classmethod
    def _deserialize_temporal_metadata(cls, metadata, net):
        """Restore temporal metadata from a saved JSON structure."""
        partition_setting = {
            'static_nodes': metadata.get('static_nodes', []),
            'dynamic_node_candidates': metadata.get('dynamic_node_candidates', []),
            'static_links': metadata.get('static_links', []),
            'dynamic_link_candidates': metadata.get('dynamic_link_candidates', []),
        }

        time_slots = metadata.get('time_slots', [])
        node_activation = {}
        for raw_t, entries in metadata.get('node_activation', {}).items():
            t = raw_t
            try:
                if raw_t in time_slots:
                    t = raw_t
                elif int(raw_t) in time_slots:
                    t = int(raw_t)
                elif float(raw_t) in time_slots:
                    t = float(raw_t)
            except (TypeError, ValueError):
                pass
            node_activation[t] = {
                cls._coerce_node_id(node_id, set(net.nodes)): value
                for node_id, value in entries
            }

        link_activation = {}
        all_links = {tuple(link) for link in net.links}
        canonical_to_link = {cls._canonical_link(link): link for link in all_links}
        for raw_t, entries in metadata.get('link_activation', {}).items():
            t = raw_t
            try:
                if raw_t in time_slots:
                    t = raw_t
                elif int(raw_t) in time_slots:
                    t = int(raw_t)
                elif float(raw_t) in time_slots:
                    t = float(raw_t)
            except (TypeError, ValueError):
                pass
            time_state = {}
            for link_data, value in entries:
                canonical_link = cls._canonical_link(link_data)
                if canonical_link in canonical_to_link:
                    time_state[canonical_to_link[canonical_link]] = value
            link_activation[t] = time_state

        temporal_setting = {
            'time_slots': time_slots,
            'node_activation': node_activation,
            'link_activation': link_activation,
            'defaults': metadata.get('defaults', {}),
        }
        for key in net._temporal_metadata_keys():
            if key in metadata:
                temporal_setting[key] = metadata[key]
        return partition_setting, temporal_setting

    def generate_topology(self, num_nodes: int, type: str = 'waxman', **kwargs):
        """
        Generate a topology for the network.

        Args:
            num_nodes (int): The number of nodes in the network.
            type (str, optional): The type of network to generate. Defaults to 'waxman'.
            **kwargs: Keyword arguments to pass to the network generator.
        """
        super().generate_topology(num_nodes, type, **kwargs)
        all_nodes = set(self.nodes)
        all_links = {tuple(link) for link in self.links}
        static_nodes, dynamic_nodes, static_links, dynamic_links = self._resolve_candidate_partitions(
            all_nodes,
            all_links,
            **kwargs,
        )
        self.static_nodes = static_nodes
        self.dynamic_node_candidates = dynamic_nodes
        self.static_links = static_links
        self.dynamic_link_candidates = dynamic_links
        self._annotate_candidate_roles()
        self.degree_benchmark = self.get_degree_benchmark()

    def generate_attrs_data(self, node: bool = True, link: bool = True) -> None:
        """
        Generate attribute data for the network.

        Args:
            node (bool, optional): Whether or not to generate node attribute data. Defaults to True.
            link (bool, optional): Whether or not to generate link attribute data. Defaults to True.
        """
        super().generate_attrs_data(node, link)
        if node:
            self._initialize_dynamic_node_parameters()
        if link:
            self._initialize_dynamic_link_parameters()
        self._initialize_activation_states()
        if node: 
            self.node_attr_benchmarks = self.get_node_attr_benchmarks()
        if link:
            self.link_attr_benchmarks = self.get_link_attr_benchmarks()
            self.link_sum_attr_benchmarks = self.get_link_sum_attr_benchmarks()

    @staticmethod
    def from_setting(setting: dict, seed: int = None) -> 'PhysicalNetwork':
        """
        Create a PhysicalNetwork object from the given setting.

        Args:
            setting (dict): The network settings.
            seed (int): The random seed for network generation.

        Returns:
            PhysicalNetwork: A PhysicalNetwork object.
        """
        setting = copy.deepcopy(setting)
        node_attrs_setting = setting.pop('node_attrs_setting')
        link_attrs_setting = setting.pop('link_attrs_setting')
        topology_setting = copy.deepcopy(setting.pop('topology'))
        partition_setting = copy.deepcopy(setting.pop('partition', {}))
        temporal_setting = PhysicalNetwork._extract_temporal_setting(setting)

        file_path = topology_setting.get('file_path')
        load_from_file = file_path not in ['', None, 'None', 'null'] and os.path.exists(file_path)

        net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, link_attrs_setting=link_attrs_setting, **setting)
        if load_from_file:
            graph = nx.read_gml(file_path, label='id')
            net._load_graph_data(graph)
            net._apply_candidate_partition_metadata(partition_setting)
            print(f'Loaded the topology from {file_path}')
        else:
            num_nodes = setting.pop('num_nodes')
            net.generate_topology(num_nodes, **topology_setting, **partition_setting)

        net._apply_temporal_metadata(temporal_setting)
        if seed is None:
            seed = setting.get('seed')
        random.seed(seed)
        np.random.seed(seed)
        net.generate_attrs_data()
        net.degree_benchmark = net.get_degree_benchmark()
        return net

    @staticmethod
    def from_topology_zoo_setting(topology_zoo_setting: dict, seed: int = None) -> 'PhysicalNetwork':
        """
        Create a PhysicalNetwork object from a topology zoo setting.

        Args:
            topology_zoo_setting (dict): A dictionary containing the setting for the physical network.
            seed (int): An optional integer value to seed the random number generators.

        Returns:
            net (PhysicalNetwork): A PhysicalNetwork object representing the physical network.
        """
        setting = copy.deepcopy(topology_zoo_setting)
        node_attrs_setting = setting.pop('node_attrs_setting')
        link_attrs_setting = setting.pop('link_attrs_setting')
        file_path = setting.pop('file_path')
        partition_setting = copy.deepcopy(setting.pop('partition', {}))
        temporal_setting = PhysicalNetwork._extract_temporal_setting(setting)

        net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, link_attrs_setting=link_attrs_setting, **setting)
        graph = nx.read_gml(file_path, label='id')
        net._load_graph_data(graph)
        net._apply_candidate_partition_metadata(partition_setting)
        net._apply_temporal_metadata(temporal_setting)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        net.generate_attrs_data()
        net.degree_benchmark = net.get_degree_benchmark()
        return net

    def save_dataset(self, dataset_dir: str) -> None:
        """
        Save the physical network dataset to a directory.

        Args:
            dataset_dir (str): The path to the directory where the physical network dataset is to be saved.
        """
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        file_path = os.path.join(dataset_dir, 'p_net.gml')
        self.to_gml(file_path)
        metadata_path = os.path.join(dataset_dir, 'p_net_metadata.json')
        write_setting(self._serialize_temporal_metadata(), metadata_path, mode='w+')

    @staticmethod
    def load_dataset(dataset_dir: str) -> 'PhysicalNetwork':
        """
        Load the physical network dataset from a directory.

        Args:
            dataset_dir (str): The path to the directory where the physical network dataset is stored.
        """
        if not os.path.exists(dataset_dir):
            raise ValueError(f'Find no dataset in {dataset_dir}.\nPlease firstly generating it.')
        file_path = os.path.join(dataset_dir, 'p_net.gml')
        p_net = PhysicalNetwork.from_gml(file_path)
        metadata_path = os.path.join(dataset_dir, 'p_net_metadata.json')
        if os.path.exists(metadata_path):
            metadata = read_setting(metadata_path)
            partition_setting, temporal_setting = PhysicalNetwork._deserialize_temporal_metadata(metadata, p_net)
            p_net._apply_candidate_partition_metadata(partition_setting)
            p_net._apply_temporal_metadata(temporal_setting)
            p_net.generate_attrs_data(node=False, link=False)
        else:
            p_net._apply_candidate_partition_metadata({})
        # get benchmark for normalization
        p_net.degree_benchmark = p_net.get_degree_benchmark()
        p_net.node_attr_benchmarks = p_net.get_node_attr_benchmarks()
        p_net.link_attr_benchmarks = p_net.get_link_attr_benchmarks()
        p_net.link_sum_attr_benchmarks = p_net.get_link_sum_attr_benchmarks()
        return p_net
