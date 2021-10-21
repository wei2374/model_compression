import matplotlib.pyplot as plt
import numpy as np


def plot_training_result(history, clr, foldername):
    plt.clf()
    plt.plot(history.history['top1'])
    plt.plot(history.history['val_top1'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(foldername+"/training_accuracy.png")
    # plt.show()

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(foldername+"/training_loss.png")
    # plt.show()

    if clr is not None:
        # plot the learning rate history
        N = np.arange(0, len(clr.history["lr"]))
        plt.figure()
        plt.plot(N, clr.history["lr"])
        plt.title("Cyclical Learning Rate (CLR)")
        plt.xlabel("Training Iterations")
        plt.ylabel("Learning Rate")
        plt.savefig(foldername+"/learning_rate.png")
        # plt.show()
