"""
Utilitary functions
"""

# Standard modules
import os


def generate_unique_logpath(logdir: str,
                            raw_run_name: str):
    """
    Find for an unused logdir logdir/raw_run_name_%d
    Arguments:
        logdir (str): the base dir
        raw_run_name (str): the prefix for the log dir
    Returns:
        a string for a path that does not exist
        following the template logdir/raw_run_name_%d
    """
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1
