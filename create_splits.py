import argparse
import glob
from imghdr import tests
import os
import random
import shutil

import numpy as np

from utils import get_module_logger


def validateDirectory(path, createIfNotExist = False):
    if not os.path.exists(path):
        if createIfNotExist:
            os.mkdir(path)
            return path
        else:
            raise Exception(f"Path does not exist: {path}")

    if not os.path.isdir(path):
        raise Exception(f"Filepath {path} is not a directory")

    return path

def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    validateDirectory(source)
    validateDirectory(destination, createIfNotExist=True)

    recordFiles = glob.glob(os.path.join(source, "*.tfrecord"))
    recordCount = len(recordFiles)

    logger.info(f"{recordCount} files.")
    random.shuffle(recordFiles)

    ## Move 10% to validation, 10% to test and the rest to train
    trainDir = validateDirectory(os.path.join(destination, "train"), True)
    valDir = validateDirectory(os.path.join(destination, "val"), True)
    testDir = validateDirectory(os.path.join(destination, "test"), True)

    testSize = 2
    logger.info(f"Test size: {testSize}")
    testFiles = recordFiles[:testSize]

    logger.info(f"Val size: {testSize}")
    valSize = 2
    valFiles = recordFiles[testSize:valSize + testSize]

    logger.info(f"Train size: {recordCount - testSize - valSize}")
    trainingFiles = recordFiles[testSize + valSize:]

    for f in testFiles:
        shutil.copyfile(f, os.path.join(testDir, os.path.basename(f)))
    for f in valFiles:
        shutil.copyfile(f, os.path.join(valDir, os.path.basename(f)))
    for f in trainingFiles:
        shutil.copyfile(f, os.path.join(trainDir, os.path.basename(f)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
