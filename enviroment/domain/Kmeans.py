# Barbara
# 开发时间：2025/4/1 21:43

from sklearn.cluster import KMeans
import numpy as np


class DroneCluster:
    def __init__(self, coverage_radius):
        self.coverage_radius = coverage_radius

    def deploy_drones(self, user_positions, num_drones):
        """返回无人机位置和用户分配"""
        kmeans = KMeans(n_clusters=num_drones)
        kmeans.fit(user_positions)

        # 用户分配 (key: user_id, value: drone_id 或 None)
        user_assignments = {}
        for user_id, pos in enumerate(user_positions):
            distances = np.linalg.norm(pos - kmeans.cluster_centers_, axis=1)
            drone_id = np.argmin(distances)
            user_assignments[user_id] = drone_id if distances[drone_id] <= self.coverage_radius else None

        return kmeans.cluster_centers_, user_assignments