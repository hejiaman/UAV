import community as community_louvain
import networkx as nx
import numpy as np
from collections import defaultdict


class DroneCluster:
    def __init__(self, coverage_radius, max_drones, beta=1.0):
        self.coverage_radius = coverage_radius
        self.max_drones = max_drones
        self.beta = beta

    def deploy_drones(self, user_positions, user_loads):
        """基于python-louvain库的优化实现"""
        # 1. 构建加权图
        G = nx.Graph()
        n_users = len(user_positions)

        # 添加节点（带权重）
        for i in range(n_users):
            G.add_node(i, weight=user_loads[i])

        # 添加边（带距离权重）
        for i in range(n_users):
            for j in range(i + 1, n_users):
                dist = np.linalg.norm(np.array(user_positions[i]) - np.array(user_positions[j]))
                if dist <= self.coverage_radius:
                    weight = 1.0 / (dist + 1e-6)
                    G.add_edge(i, j, weight=weight)

        # 2. 运行Louvain算法
        partition = community_louvain.best_partition(G, resolution=self.beta)

        # 3. 合并社区直到满足无人机数量限制
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)

        # 如果社区数量超过无人机数量限制
        while len(communities) > self.max_drones:
            # 找到两个最小社区合并
            comm_sizes = {k: sum(user_loads[u] for u in v) for k, v in communities.items()}
            smallest_comms = sorted(comm_sizes.items(), key=lambda x: x[1])[:2]

            # 合并社区
            comm_to_keep, comm_to_merge = smallest_comms[0][0], smallest_comms[1][0]
            communities[comm_to_keep].extend(communities[comm_to_merge])
            del communities[comm_to_merge]

            # 更新分区
            for node in communities[comm_to_keep]:
                partition[node] = comm_to_keep

        # 4. 计算无人机位置（加权中心）
        drone_positions = []
        assignments = {}

        for comm_id, members in communities.items():
            # 计算加权中心
            total_load = sum(user_loads[u] for u in members)
            if total_load == 0:
                center = np.mean([user_positions[u] for u in members], axis=0)
            else:
                center = np.sum(
                    [np.array(user_positions[u]) * user_loads[u] for u in members],
                    axis=0
                ) / total_load

            drone_positions.append(center.tolist())

            # 记录分配
            for u in members:
                assignments[u] = len(drone_positions) - 1

        return drone_positions, assignments