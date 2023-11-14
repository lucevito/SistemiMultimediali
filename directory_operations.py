import os
import random
import shutil


def list_leaf_folders(path):
    """
    Returns a list of all leaf folders in a specified path.

    Args:
        path (str): Search path.

    Returns:
        list: List of leaf folders in the specified path.
    """
    leaf_folders = []

    for root, dirs, files in os.walk(path):
        if not dirs:
            leaf_folders.append(root)

    return leaf_folders


def create_directory(path):
    """
    Creates a directory if it does not exist.

    Args:
        path (str): Path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
