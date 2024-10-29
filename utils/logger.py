import logging

logging.basicConfig(level=logging.INFO)


def log_metrics(metric_name, metric_value):
    logging.info(f"{metric_name}: {metric_value}")
