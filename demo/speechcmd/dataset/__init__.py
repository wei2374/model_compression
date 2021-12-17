import os

data_folder = "SpeechCommands"
tmp_dir = "/home/wei/Documents/workspace/TensorNAS/TensorNAS/Demos/Datasets/tmp"
data_dir = os.path.join(tmp_dir, data_folder)


def GetData():

    from .kws_util import parse_command
    from .get_dataset import get_training_data

    Flags, unparsed = parse_command()
    Flags.data_dir = data_dir

    ds_train, ds_test, ds_val = get_training_data(Flags)

    train_len = len(ds_train)
    val_len = len(ds_val)
    test_len = len(ds_test)

    import tensorflow as tf

    shape = tuple(tf.compat.v1.data.get_output_shapes(ds_train)[0].as_list()[1:])

    return ds_train, ds_val, ds_test, shape, train_len, val_len, test_len
