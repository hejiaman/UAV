CONFIG_TRAINING = {
    'num_episodes': 1000,
    'learning_rate': 0.001,
    'gamma': 0.99,
}

CONFIG_ENVIRONMENT = {
    'grid_size': (10, 10),
    'num_obstacles': 5,
    'start_position': (0, 0),
}

CONFIG_SENSING_DEVICE = {

}

CONFIG_TRANSMISSION = {

}

CONFIG_AP = {

}

CONFIG_DATA = {
    'data_mapping': {
            'image': True,
            'text': False,
            'audio': True,
        }
}