# Barbara
# 开发时间：2025/4/1 15:45
import numpy as np
import networkx as nx
import community as community_louvain  # python-louvain库


class LouvainDeployer:
    def __init__(self, config):
        """
        基于Louvain模块度的无人机部署器

        参数:
            config: 包含以下键的配置字典
                - user_positions: 用户位置列表 [[x1,y1], [x2,y2], ...]
                - user_demand: 用户需求列表 [d1, d2, ...] (可选)
                - num_drones: 无人机数量
                - area_size: 区域大小 [width, height]
                - comm_threshold: 用户连接阈值(米)
        """
        self.user_positions = np.array(config['user_positions'])
        self.user_demand = config.get('user_demand', np.ones(len(config['user_positions'])))
        self.num_drones = config['num_drones']
        self.area_size = config['area_size']
        self.threshold = config.get('comm_threshold', 300)

    def build_community_graph(self):
        """
        构建用户社区图
        返回: networkX Graph对象
        """
        G = nx.Graph()

        # 添加节点(带需求属性)
        for i, pos in enumerate(self.user_positions):
            G.add_node(i, pos=pos, demand=self.user_demand[i])

        # 添加边(基于距离阈值)
        for i in range(len(self.user_positions)):
            for j in range(i + 1, len(self.user_positions)):
                distance = np.linalg.norm(self.user_positions[i] - self.user_positions[j])
                if distance <= self.threshold:
                    weight = 1 - distance / self.threshold  # 距离越近权重越高
                    G.add_edge(i, j, weight=weight)

        return G

    def deploy_drones(self):
        """
        执行Louvain聚类并计算无人机部署位置

        返回:
            drone_positions: 无人机部署位置列表 [(x1,y1), (x2,y2), ...]
            communities: 社区划分结果 {社区ID: [用户节点列表]}
        """
        G = self.build_community_graph()

        # 执行Louvain聚类
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)

        # 将分区格式转换为 {社区ID: [节点列表]}
        communities = {}
        for node, community_id in partition.items():
            communities.setdefault(community_id, []).append(node)

        # 合并社区直到数量<=无人机数量
        while len(communities) > self.num_drones:
            self._merge_smallest_communities(G, communities)

        # 计算每个社区的质心作为无人机位置
        drone_positions = []
        for com_id, nodes in communities.items():
            positions = [self.user_positions[n] for n in nodes]
            demands = [self.user_demand[n] for n in nodes]

            # 加权质心(考虑需求)
            centroid = np.average(positions, axis=0, weights=demands)
            drone_positions.append(centroid.tolist())

        # 确保位置在合法范围内
        drone_positions = np.clip(drone_positions, [0, 0], self.area_size)

        return drone_positions, communities

    def _merge_smallest_communities(self, G, communities):
        """合并最小的两个社区"""
        # 找到两个最小的社区
        sorted_coms = sorted(communities.items(), key=lambda x: len(x[1]))
        com1_id, com1_nodes = sorted_coms[0]
        com2_id, com2_nodes = sorted_coms[1]

        # 合并社区
        communities[com1_id].extend(com2_nodes)
        del communities[com2_id]

        # 更新分区(为了可视化等后续使用)
        for node in com2_nodes:
            G.nodes[node]['community'] = com1_id

    def visualize_communities(self, communities):
        """可视化社区划分结果(需要matplotlib)"""
        import matplotlib.pyplot as plt

        G = self.build_community_graph()
        pos = nx.get_node_attributes(G, 'pos')

        plt.figure(figsize=(10, 8))

        # 绘制节点(按社区着色)
        for com_id, nodes in communities.items():
            nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                                   node_color=np.random.rand(3, ),
                                   label=f'Community {com_id}')

        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.2)

        plt.title('User Communities by Louvain Method')
        plt.legend()
        plt.show()