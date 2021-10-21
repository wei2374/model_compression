import matplotlib.pyplot as plt
import numpy as np


def plot_rank_percentage(layer_ranks, method, foldername):
    x_axis = []
    input_p = []
    output_p = []
    width = 0.35

    for layer_id in layer_ranks:
        if(len(layer_ranks[layer_id]) == 4):
            x_axis.append(f"{layer_id}")
            input_percentage = layer_ranks[layer_id][2]/layer_ranks[layer_id][1]
            output_percentage = layer_ranks[layer_id][3]/layer_ranks[layer_id][0]
            input_p.append(input_percentage)
            output_p.append(output_percentage)

        else:
            x_axis.append(f"{layer_id}")
            input_percentage = layer_ranks[layer_id][2]/layer_ranks[layer_id][0]
            output_p.append(input_percentage)

    ind = np.arange(len(layer_ranks.keys()))
    if method == "tucker2D" or method == "channel_all":
        plt.bar(ind, input_p, width, label='Input')
    if method == "channel_output" or method == "channel_nl":
        plt.bar(ind, input_p, width, label='Input')
        plt.bar(ind + width, output_p, width, label='Output')
    plt.bar(ind + width, output_p, width, label='Output')
    plt.xticks(ind + width / 2, x_axis)

    plt.xlabel('layer id')
    plt.ylabel('percentage')
    plt.title(f"Rank estimation")
    plt.legend()
    filename = foldername+f"/rank_estimation.png"
    plt.ylim((0, 1))
    plt.savefig(filename)
    # plt.show()
