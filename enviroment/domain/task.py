# Barbara
# 开发时间：2025/4/1 21:42

class Task:
    def __init__(self, task_dict):
        """
        从字典初始化任务对象
        Args:
            task_dict: 包含任务属性的字典
        """
        self.__dict__.update(task_dict)
        self.initial_compute = task_dict['compute_req']  # 保留初始计算量

    def __repr__(self):
        return f"Task(user={self.user_id}, status={self.status}, compute_left={self.compute_req:.1f}MIPS)"