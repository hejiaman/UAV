import random

config = {
    'training': {
        'num_episodes': 1000,
        'learning_rate': 0.001,
        'gamma': 0.99,
    },
    'environment': {
        'time_slot': 1,
        'time_unit': 10
    },
    'devices': {
        'num_devices': 10,
    },
    'sensing_device': {
        'num_devices': 3,
        'device_types': ['type1', 'type2', 'type3'],
        'positions': [[0, 1], [1, 1], [1, 0]],
        'computing_capability': lambda: random.uniform(2, 5)
        # 'positions': lambda: [[random.randint(0, 10), random.randint(0, 10)] for _ in range(3)]
    },
    'ap': {
        'num_devices': 1,
        'positions': [0, 0],
        'computing_capability': lambda: random.uniform(20, 30)
    },
    'data': {
        'data_type_mapping': {
            'type1': 'type1',
            'type2': 'type2',
            'type3': 'type3',
        },
        'data_process_mapping': {
            'type1': True,
            'type2': False,
            'type3': True,
        },
        'data_size_mapping': {
            'type1': 10,
            'type2': 1,
            'type3': 3,
        },
    },
    'sensing_task': {
        # 'sensing_duration': lambda: random.randint(0, 3)
        'sensing_duration': 3  # 固定感知时间, time_slot
    },
    'transmission_task': {
        # Add transmission-related parameters here if needed (e.g., bandwidth, power)
        # 取整得到time_slot
    },
    'process_task': {
        # 取整得到time_slot
    }
}
