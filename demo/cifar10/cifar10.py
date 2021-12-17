dataset_name = "Cifar10"
dataset_zip = "cifar-10-python.tar.gz"
dataset_url = "https://www.cs.toronto.edu/~kriz/{}".format(dataset_zip)
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data

import pathlib, os

tmp_dir = str(pathlib.Path(__file__).parent.resolve()) + "/tmp"
zip_dir = tmp_dir + "/zips"


def _make_tmp_dir():

    if os.path.isdir(tmp_dir) == False:
        os.mkdir(tmp_dir)

    if os.path.isdir(zip_dir) == False:
        os.mkdir(zip_dir)


def make_dataset_dirs(dataset_name):

    _make_tmp_dir()

    dir = tmp_dir + "/{}".format(dataset_name)

    if os.path.isdir(dir) == False:
        os.mkdir(dir)


def bar_progress(current, total, width=80):
    import sys

    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

def GetData():
    import wget, os

    make_dataset_dirs(dataset_name)

    if not os.path.isfile(os.path.join(zip_dir, dataset_zip)):
        print("Downloading Cifar10 dataset tar")
        wget.download(dataset_url, out=zip_dir, bar=bar_progress)
    else:
        print("Cifar10 tar already exists, skipping download")

    output_dir = os.path.join(tmp_dir, dataset_name)

    if not len(os.listdir(output_dir)):
        import tarfile

        print("Extracting Cifar10 tar")
        tar = tarfile.open(zip_dir + "/{}".format(dataset_zip), "r:gz")
        tar.extractall(path=output_dir)
        tar.close()

        import shutil

        out_files = os.listdir(output_dir)

        sub_out_files = os.listdir(os.path.join(output_dir, out_files[0]))

        for file in sub_out_files:
            shutil.move(os.path.join(output_dir, out_files[0], file), output_dir)

        shutil.rmtree(os.path.join(output_dir, out_files[0]))
    else:
        print("Cifar10 tar already extracted")

    (
        train_data,
        train_filenames,
        train_labels,
        test_data,
        test_filenames,
        test_labels,
        label_names,
    ) = load_cifar_10_data(output_dir)

    return test_data, train_data, test_labels, train_labels, test_data[0].shape

def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b"label_names"]
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b"data"]
        else:
            cifar_train_data = np.vstack(
                (cifar_train_data, cifar_train_data_dict[b"data"])
            )
        cifar_train_filenames += cifar_train_data_dict[b"filenames"]
        cifar_train_labels += cifar_train_data_dict[b"labels"]

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b"data"]
    cifar_test_filenames = cifar_test_data_dict[b"filenames"]
    cifar_test_labels = cifar_test_data_dict[b"labels"]

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return (
        cifar_train_data,
        cifar_train_filenames,
        to_categorical(cifar_train_labels),
        cifar_test_data,
        cifar_test_filenames,
        to_categorical(cifar_test_labels),
        cifar_label_names,
    )