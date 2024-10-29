config = {
    'training': {  # Better organization
        'num_episodes': 1000,
        'learning_rate': 0.001,
        'gamma': 0.99,
    },
    'environment': {  # Remove unused parameters for now
        # 'grid_size': (10, 10),  # Remove or use
        # 'num_obstacles': 5,       # Remove or use
        # 'start_position': (0, 0), # Remove or use
    },
    'sensing_device': {
        'num_devices': 3,
        'device_types': ['type1', 'type2', 'type3'],
        'positions': [[0, 1], [1, 1], [1, 0]]
    },
    'transmission': {
        # Add transmission-related parameters here if needed (e.g., bandwidth, power)
    },
    'ap': {
        'num_devices': 3,
        'positions': [[0, 0]]
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
}