import gym
import numpy as np


from enviroment.domain.user import User
from enviroment.domain.uav import Drone
from enviroment.domain.Kmeans import DroneCluster
# from enviroment.domain.Louvain import DroneCluster
from utils.metrics import MetricsCalculator

import os
os.environ["OMP_NUM_THREADS"] = "1"



class DroneSchedulingEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_step = 0
        self._init_environment()
        self.coverage_radius = 50  # 根据你想要的覆盖半径来设定，单位跟位置保持一致

        # 动作空间：每个无人机对当前任务的选择
        # 0:本地执行, 1~N:转发给对应无人机
        self.action_space = gym.spaces.MultiDiscrete(
        [self.config['num_drones'] + 1] * self.config['num_drones'])

        '''
        -----可以改进（改编成连续动作空间，比如转发概率）)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(num_drones,))
        让每个无人机的动作值表示：
        0~0.5：本地执行
        0.5~1.0：转发，具体目标无人机用 softmax 选择
        '''

        # 状态空间：[队列长度, CPU利用率, 剩余时延] × num_drones
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf,
            shape=(self.config['num_drones'] * 3,),
            dtype=np.float32)

        # 优化方向：队列长度queue_length / max_queue_length和时延remaining_delay / max_delay可以做一个归一化


    def _init_environment(self):
        """初始化用户和无人机"""
        self.users = [User(i, pos, self.config['task_rate'])
                      for i, pos in enumerate(self.config['user_positions'])]

        # 获取用户数据
        user_positions = [user.position for user in self.users]
        user_loads = [user.task_rate for user in self.users]

        # 使用优化部署
        # 使用 KMeans 聚类部署
        deployer = DroneCluster(
            coverage_radius=self.config['coverage_radius']
        )

        drone_positions, assignments = deployer.deploy_drones(
            user_positions=user_positions,
            num_drones=self.config['num_drones']
        )

        # print(f"Number of drone positions: {len(drone_positions)}")
        # print(f"Number of assignments: {len(assignments)}")

        # 初始化无人机
        self.drones = [
            Drone(i, pos, self.config)
            for i, pos in enumerate(drone_positions)
        ]

        # 部署无人机后添加负载初始化
        for drone in self.drones:
            drone.current_load = 0  # 确保所有无人机都有此属性

        # 绑定用户时安全累加
        for user_id, drone_id in assignments.items():
            if hasattr(self.drones[drone_id], 'current_load'):  # 安全检查
                self.drones[drone_id].current_load += self.users[user_id].task_rate
            else:
                self.drones[drone_id].current_load = self.users[user_id].task_rate

        self.completed_tasks = []


    def step(self, actions):
        """执行一个时间步"""
        self.current_step += 1

        # 1. 处理当前任务
        for drone in self.drones:
            if drone.process_task():  # 任务完成
                self.completed_tasks.append(drone.current_task)
                drone.current_task = None

        # 2. 执行调度决策
        self._execute_actions(actions)

        # 打印动作执行后的状态
        print(f"\n[Step {self.current_step}] After executing actions:")
        for drone in self.drones:
            status = drone.current_task['status'] if drone.current_task else "None"
            print(f"Drone {drone.id}: current_task={status}, queue_len={len(drone.task_queue)}")


        # 3. 生成新任务
        self._generate_tasks()

        # 打印任务生成情况
        print(f"\n[Step {self.current_step}] After generating tasks:")
        for user in self.users:
            assigned = user.assigned_drone
            print(
                f"User {user.id}: assigned to Drone {assigned}, task queue = {len(self.drones[assigned].task_queue) if assigned is not None else 'N/A'}")

        # 获取状态和奖励
        next_state = self._get_state()
        reward = MetricsCalculator.calculate_reward(self)
        done = self.current_step >= self.config['max_steps']

        print(f"[Step {self.current_step}] Completed Tasks This Step: {len(self.completed_tasks)}")
        for t in self.completed_tasks:
            print(f"Task {t.user_id} finished at {t.complete_time}, delay: {t.complete_time - t.create_time}")

        return next_state, reward, done, self._get_metrics()

    def _execute_actions(self, actions):
        """处理调度动作"""
        for drone_id, action in enumerate(actions):
            if drone_id >= len(self.drones):
                print(f"Error: Drone ID {drone_id} is out of range. Total drones: {len(self.drones)}")
                continue  # 跳过错误的动作

            drone = self.drones[drone_id]

            # 只对新到达的 pending 任务做决策
            if drone.current_task and drone.current_task['status'] == 'pending':
                if action == 0:  # 本地执行
                    drone.current_task['status'] = 'computing'
                else:  # 转发
                    target_id = action - 1
                    if 0 <= target_id < len(self.drones) and target_id != drone_id:
                        target_drone = self.drones[target_id]
                        drone.forward_task(target_drone, drone.current_task)
                    else:
                        print(f"Invalid forwarding target: {target_id}, fallback to local execution.")
                        drone.current_task['status'] = 'computing'

    def _generate_tasks(self):
        """用户生成新任务"""
        for user in self.users:
            # 先重置当前的关联（可选）
            user.assigned_drone = None

            # 检查有哪些无人机覆盖了该用户
            print(f"[Check Coverage] User {user.id} at {user.position}")
            for drone in self.drones:
                dist = np.linalg.norm(np.array(user.position) - np.array(drone.position))
                in_range = dist <= self.coverage_radius
                print(f"    ↳ Drone {drone.id} at {drone.position}, dist = {dist:.2f}, in range: {in_range}")
                if in_range:
                    # 将用户分配给第一个在覆盖范围内的无人机
                    user.assigned_drone = drone.id
                    print(f"→ User {user.id} assigned to Drone {drone.id}")
                    break  # 如果你只想分配给一个无人机，就break；否则可以用策略选一个最优的

            # 生成任务并添加进队列
            task = user.generate_task(self.current_step)
            if task and user.assigned_drone is not None:
                self.drones[user.assigned_drone].task_queue.append(task)
                print(f"+++ Task from User {user.id} added to Drone {user.assigned_drone}'s queue")

    def _get_state(self):
        state = []   # 每个无人机固定贡献3个特征
        for drone in self.drones:
            state.extend([
                len(drone.task_queue) / self.config['max_queue_length'],
                drone.current_load / 100.0,
                1 if drone.current_task else 0
            ])
        # 添加2个全局特征
        state.extend([
            len(self.completed_tasks) / 100,
            self.current_step / self.config['max_steps']
        ])
        expected_dim = 3 * len(self.drones) + 2
        assert len(state) == expected_dim, f"状态维度应为{expected_dim}，实际{len(state)}"
        return np.array(state, dtype=np.float32)


    def _get_metrics(self):
        """获取评估指标"""
        return {
            'completed': len(self.completed_tasks),
            'avg_delay': MetricsCalculator.average_delay(self.completed_tasks),
            'total_energy': sum(d.energy_consumed for d in self.drones),
            'load_balance': MetricsCalculator.load_balance(self.drones)
        }

    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.completed_tasks = []
        self._init_environment()
        return self._get_state()