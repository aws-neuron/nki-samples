"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Kernels written by the AWS Neuron.

WARNING: These kernels:
   - Are tested only against internal nightly builds
   - May not be compatible with public NeuronSDK releases
   - Have not been extensively tested across all input configurations
   - Carry no compatibility guarantees
   - The behavior of these kernels may be modified without prior notice
"""

from typing import List, Dict, Tuple, Sequence, Set
from dataclasses import dataclass, field

REPLICA_GROUP_DTYPE = List[int]
REPLICA_GROUPS_DTYPE = List[REPLICA_GROUP_DTYPE]
SRC_TGT_PAIR_DTYPE = Tuple[int, int]
SRC_TGT_PAIRS_DTYPE = List[SRC_TGT_PAIR_DTYPE]


@dataclass(frozen=True)
class CollectivesConfig:
    """
    From src_tgt_pairs, induce:
    - all_ranks
    - rank_src_tgt_pairs
    - rank_sorted_replica_groups

    - send_replica_groups
    - recv_replica_groups
    - rank_send_replica_groups
    - rank_recv_replica_groups
    """
    
    src_tgt_pairs: SRC_TGT_PAIRS_DTYPE = field(default_factory=list)
    replica_groups: REPLICA_GROUPS_DTYPE = field(default_factory=list)
    all_ranks: List[int] = field(init=False)
    rank_src_tgt_pairs: Dict[int, List[Tuple[int, int]]] = field(init=False)
    rank_sorted_replica_groups: Dict[int, List[List[int]]] = field(init=False)
    send_replica_groups: List[List[int]] = field(init=False)
    recv_replica_groups: List[List[int]] = field(init=False)
    rank_send_replica_groups: Dict[int, List[int]] = field(init=False)
    rank_recv_replica_groups: Dict[int, List[int]] = field(init=False)

    def __post_init__(self):
        if self.src_tgt_pairs and self.replica_groups:
            raise ValueError(
                "Only one of src_tgt_pairs or replica_groups should be provided."
            )
        if self.replica_groups:
            src_tgt_pairs = self._replica_groups_to_src_tgt_pairs()
        else:
            src_tgt_pairs = self.src_tgt_pairs

        sorted_src_tgt_pairs = _sort_src_tgt_pairs(src_tgt_pairs)
        object.__setattr__(self, 'src_tgt_pairs', sorted_src_tgt_pairs)
        self._check_valid_inputs()

        object.__setattr__(self, 'all_ranks', self._get_all_ranks())
        
        rank_src_tgt_pairs, rank_sorted_replica_groups = self._filter_rank_src_tgt_pairs()
        object.__setattr__(self, 'rank_src_tgt_pairs', rank_src_tgt_pairs)
        object.__setattr__(self, 'rank_sorted_replica_groups', rank_sorted_replica_groups)

        send_replica_groups, recv_replica_groups = self._find_replica_groups()
        object.__setattr__(self, 'send_replica_groups', send_replica_groups)
        object.__setattr__(self, 'recv_replica_groups', recv_replica_groups)

        rank_send_replica_groups, rank_recv_replica_groups = self._find_rank_replica_groups()
        object.__setattr__(self, 'rank_send_replica_groups', rank_send_replica_groups)
        object.__setattr__(self, 'rank_recv_replica_groups', rank_recv_replica_groups)

    def _check_valid_inputs(self):
        """
        Check if the input src_tgt_pairs are valid.
        """
        for pair in self.src_tgt_pairs:
            assert len(pair) == 2 and all(
                [type(x) is int for x in pair]
            ), f"src_tgt_pair {pair} is not a pair of int."

    def _get_all_ranks(self) -> REPLICA_GROUP_DTYPE:
        """Get all the ranks involved in all replica_groups

        Returns:
            REPLICA_GROUP_DTYPE: group of all ranks
        """
        all_ranks = set()
        for src_tgt_pair in self.src_tgt_pairs:
            all_ranks.add(src_tgt_pair[0])
            all_ranks.add(src_tgt_pair[1])
        all_ranks = sorted(list(all_ranks))
        return all_ranks

    def _filter_rank_src_tgt_pairs(
        self,
    ) -> Tuple[Dict[int, SRC_TGT_PAIRS_DTYPE], Dict[int, REPLICA_GROUP_DTYPE]]:
        """
        Filter the src_tgt_pairs and sorted_replica_group relevant for each rank
        replica_groups: [[rank_0, rank_1, ...], [rank_2, rank_3, ...], ...]
        Indicates groups of ranks separated by disjoint communication.

        Args:
            rank_id (int): rank ID

        Returns:
            Tuple[Dict[int, SRC_TGT_PAIRS_DTYPE], Dict[int, REPLICA_GROUP_DTYPE]]:
            src_tgt_pairs of each rank,
            sorted_replica_groups of each rank, does not represent communication relation
        """
        graph = {}
        for source, target in self.src_tgt_pairs:
            if source not in graph:
                graph[source] = []
            graph[source].append(target)
        all_src_tgt_pairs = {}
        sorted_replica_groups = {}
        for rank_id in self.all_ranks:
            rank_src_tgt_pairs = []
            rank_replica_group = set()
            _dfs(graph, rank_id, rank_replica_group, rank_src_tgt_pairs)
            rank_src_tgt_pairs = _sort_src_tgt_pairs(rank_src_tgt_pairs)
            all_src_tgt_pairs[rank_id] = rank_src_tgt_pairs
            sorted_replica_groups[rank_id] = sorted(list(rank_replica_group))
        return all_src_tgt_pairs, sorted_replica_groups

    def _find_replica_groups(
        self,
    ) -> Tuple[REPLICA_GROUPS_DTYPE, REPLICA_GROUPS_DTYPE]:
        """
        Find the send/recv_replica_groups

        Each send_replica_group has a "send to" relation,
        i.e. send_replica_group[i] is sending to send_replica_group[i+1].

        Each recv_replica_group has a "receive from" relation,
        i.e. recv_replica_group[i] is receiving from recv_replica_group[i+1].

        Different groups are independent of each other and their orders are interchangeable.
        However, ranks within each group have a rotation-invariant order.
        """
        adj_list = {src: tgt for src, tgt in self.src_tgt_pairs}
        send_replica_groups = []
        recv_replica_groups = []
        visited = set()
        for src in adj_list:
            if src not in visited:
                replica_group = []
                current = src
                while current not in visited:
                    visited.add(current)
                    replica_group.append(current)
                    current = adj_list[current]
                send_replica_groups.append(replica_group)
                recv_replica_groups.append(replica_group[::-1])
        send_replica_groups = _sort_replica_groups(send_replica_groups)
        recv_replica_groups = _sort_replica_groups(recv_replica_groups)
        return send_replica_groups, recv_replica_groups

    def _find_rank_replica_groups(
        self,
    ) -> Tuple[Dict[int, REPLICA_GROUP_DTYPE], Dict[int, REPLICA_GROUP_DTYPE]]:
        """
        Compute the send/recv replica groups for each rank
        Rotate each replica group by placing the rank first.

        Returns:
            Dict[int, Dict[str, REPLICA_GROUP_DTYPE]]: top-layer keys are rank IDs, second-layer keys are "send_to" and "recv_from"
        """
        rank_send_replica_groups = {}
        for send_replica_group in self.send_replica_groups:
            for rank in send_replica_group:
                rank_send_replica_group = _rotate_replica_group(
                    send_replica_group, rank
                )
                rank_send_replica_groups[rank] = rank_send_replica_group

        rank_recv_replica_groups = {}
        for recv_replica_group in self.recv_replica_groups:
            for rank in recv_replica_group:
                rank_recv_replica_group = _rotate_replica_group(
                    recv_replica_group, rank
                )
                rank_recv_replica_groups[rank] = rank_recv_replica_group
        return rank_send_replica_groups, rank_recv_replica_groups
    
    def _replica_groups_to_src_tgt_pairs(self) -> SRC_TGT_PAIRS_DTYPE:
        """Convert replica groups to src_tgt_pairs.

        Args:
            replica_groups (REPLICA_GROUPS_DTYPE): replica_groups

        Returns:
            SRC_TGT_PAIRS_DTYPE: src_tgt_pairs
        """
        src_tgt_pairs = []
        for replica_group in self.replica_groups:
            for i in range(len(replica_group)):
                src_tgt_pairs.append(
                    (replica_group[i], replica_group[(i + 1) % len(replica_group)])
                )
        return src_tgt_pairs


def _dfs(
    graph: Dict[int, List[int]],
    start_node: int,
    visited_nodes: Set,
    visited_edges: SRC_TGT_PAIRS_DTYPE,
):
    """
    Performs a Depth-First Search (DFS) on the given directed graph starting from the specified node.

    Args:
        graph (Dict[int, List[int]]): A dictionary representing the graph, where the keys are nodes
            and the values are lists of neighboring nodes.
        start_node (int): The node to start the DFS from.
        visited_nodes (Set): A set to keep track of visited nodes during the DFS.
        visited_edges (SRC_TGT_PAIRS_DTYPE): A list to store the visited edges during the DFS.
            Each edge is represented as a tuple (source, target).

    This function performs a DFS traversal on the given graph starting from the `start_node`.
    It marks the visited nodes in the `visited_nodes` set and appends the visited edges to the
    `visited_edges` list. The DFS recursively explores the neighbors of each visited node.
    """
    if start_node in visited_nodes:
        return
    visited_nodes.add(start_node)
    for neighbor in graph[start_node]:
        edge = (start_node, neighbor)
        visited_edges.append(edge)
        _dfs(graph, neighbor, visited_nodes, visited_edges)


def _rotate_replica_group(
    replica_group: REPLICA_GROUP_DTYPE, first_rank: int
) -> REPLICA_GROUP_DTYPE:
    """Rotate the replica group by having the given first_rank first.

    Args:
        replica_group (REPLICA_GROUP_DTYPE): unrotated replica_group
        first_rank (int): the first rank in the rotated replica_group

    Returns:
        REPLICA_GROUP_DTYPE: rotated replica_group
    """
    first_idx = replica_group.index(first_rank)
    return list(replica_group[first_idx:]) + list(replica_group[:first_idx])


def _sort_replica_groups(replica_groups: REPLICA_GROUPS_DTYPE) -> REPLICA_GROUPS_DTYPE:
    """
    Rotate each replica group by having the smallest rank first.
    Sort replica groups by the first rank in each group.

    Args:
        replica_groups (REPLICA_GROUPS_DTYPE): unsorted replica_groups

    Returns:
        REPLICA_GROUPS_DTYPE: sorted replica_groups
    """
    sorted_replica_groups = []
    for replica_group in replica_groups:
        rotated_replica_group = _rotate_replica_group(replica_group, min(replica_group))
        sorted_replica_groups.append(rotated_replica_group)
    sorted_groups = sorted(sorted_replica_groups, key=lambda x: x[0])
    return sorted_groups


def _sort_src_tgt_pairs(src_tgt_pairs: SRC_TGT_PAIRS_DTYPE) -> SRC_TGT_PAIRS_DTYPE:
    """Sort pairs by source first, then target.

    Args:
        src_tgt_pairs (SRC_TGT_PAIRS_DTYPE): unsorted src_tgt_pairs

    Returns:
        SRC_TGT_PAIRS_DTYPE: sorted src_tgt_pairs
    """
    return sorted(src_tgt_pairs, key=lambda x: (x[0], x[1]))
