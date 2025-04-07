import gym
import numpy as np
from enviroment.domain.user import User
from enviroment.domain.drone import Drone
from enviroment.domain.Kmeans import DroneCluster
from enviroment.domain.Louvain import DroneCluster
from utils.metrics import MetricsCalculator


class DroneSchedulingEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_step = 0
        self._init_environment()


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
        deployer = DroneCluster(
            coverage_radius=self.config['coverage_radius'],
            max_drones=self.config['num_drones'],
            beta=self.config.get('beta', 1.0)
        )

        drone_positions, assignments = deployer.deploy_drones(
            user_positions=user_positions,
            user_loads=user_loads
        )

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

        # # 聚类部署无人机
        # cluster = DroneCluster(self.config['coverage_radius'])
        # drone_positions, assignments = cluster.deploy_drones(
        #     self.config['user_positions'],
        #     self.config['num_drones'])
        #
        # # 绑定用户到无人机
        # for user_id, drone_id in assignments.items():
        #     if drone_id is not None:
        #         self.users[user_id].assigned_drone = drone_id
        #
        # # 初始化无人机
        # self.drones = [Drone(i, pos, self.config)
        #                for i, pos in enumerate(drone_positions)]
        # self.completed_tasks = []

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

        # 3. 生成新任务
        self._generate_tasks()

        # 获取状态和奖励
        next_state = self._get_state
        reward = MetricsCalculator.calculate_reward(self)
        done = self.current_step >= self.config['max_steps']

        return next_state, reward, done, self._get_metrics()

    def _execute_actions(self, actions):
        """处理调度动作"""
        for drone_id, action in enumerate(actions):
            drone = self.drones[drone_id]

            # 只对新到达的pending任务做决策
            if drone.current_task and drone.current_task['status'] == 'pending':
                if action == 0:  # 本地执行
                    drone.current_task['status'] = 'computing'
                else:  # 转发
                    target_id = action - 1
                    if target_id < len(self.drones):
                        self._forward_task(drone, self.drones[target_id])

    def _generate_tasks(self):
        """用户生成新任务（需要设置初始状态）"""
        for user in self.users:
            if user.assigned_drone is not None:
                task = user.generate_task(self.current_step)
                if task:
                    task['status'] = 'pending'  # 新增：明确设置初始状态
                    self.drones[user.assigned_drone].add_task(task)

    def _generate_tasks(self):
        """用户生成新任务"""
        for user in self.users:
            task = user.generate_task(self.current_step)
            if task and user.assigned_drone is not None:
                self.drones[user.assigned_drone].task_queue.append(task)

    # def _get_state(self):
    #     """获取环境状态"""
    #     state = []
    #     for drone in self.drones:
    #         # 队列长度（归一化）
    #         queue_len = len(drone.task_queue) / self.config['max_queue_length']
    #
    #         # CPU利用率（当前任务进度）
    #         progress = 0
    #         if drone.current_task and drone.current_task['status'] == 'computing':
    #             progress = drone.current_task['compute_req'] / drone.current_task['initial_compute']
    #
    #         # 最紧急任务的剩余时延
    #         urgency = 0
    #         if drone.task_queue:
    #             nearest_deadline = min(
    #                 (t['deadline'] - self.current_step) / self.config['max_deadline']
    #                 for t in drone.task_queue)
    #             urgency = max(0, nearest_deadline)
    #
    #         state.extend([queue_len, progress, urgency])
    #     return np.array(state)

    # # PPO:修改get_state()返回归一化状态
    # def _get_state(self):
    #     state = []
    #     for drone in self.drones:
    #         queue_ratio = len(drone.task_queue) / self.config['max_queue_length']
    #         cpu_util = drone.current_task['compute_req'] / drone.current_task[
    #             'initial_compute'] if drone.current_task else 0
    #         state.extend([queue_ratio, cpu_util])
    #     return np.array(state, dtype=np.float32)
    def _get_state(self):
        state = []
        # 每个无人机固定贡献3个特征
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
        state = np.array(state, dtype=np.float32)
        print(f"状态向量生成: 无人机数={len(self.drones)}, 总维度={len(state)}")  # 调试
        return state


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
        return self._get_state