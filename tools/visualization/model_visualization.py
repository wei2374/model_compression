from tools.profilers.net_flops import net_flops
import matplotlib.pyplot as plt
import tensorflow as tf

# define nodeType Leaf node, distinguish node, definition of arrow type

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_layer_index(dic, layer_target):
    for index, layer in dic.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if layer == layer_target:
            return index


def plotNode(nodeText, centerPt, parentPt, nodeType, ax):
    ax.annotate(nodeText, xy=parentPt, xycoords='data', xytext=centerPt, textcoords='data',
                va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)
    # This parameter is a bit scary. did not understand


def plot_sequence(layer, old_end_point, ax, pruned_ratio, layer_index_dic):
    while len(layer.outbound_nodes) == 1 and len(layer.inbound_nodes[0].flat_input_ids) == 1:
        layer_index = get_layer_index(layer_index_dic, layer)
        if isinstance(layer, tf.keras.layers.Conv2D) and pruned_ratio is not None and\
             layer_index in pruned_ratio.keys():
            text = f'{layer.name} + prune ratio is {pruned_ratio[layer_index]}'
        else:
            text = f'{layer.name}'
        layer = layer.outbound_nodes[0].layer
        start_point = (old_end_point[0], old_end_point[1]-0.05)
        end_point = (old_end_point[0], old_end_point[1]-0.2)
        plotNode(text, end_point, start_point, leafNode, ax)
        old_end_point = end_point
    if len(layer.outbound_nodes) == 2:
        if isinstance(layer, tf.keras.layers.Conv2D) and pruned_ratio is not None:
            layer_index = get_layer_index(layer_index_dic, layer)
            text = f'{layer.name} + prune ratio is {pruned_ratio[layer_index]}'
        else:
            text = f'{layer.name}'
        start_point = (old_end_point[0], old_end_point[1]-0.05)
        end_point = (old_end_point[0], old_end_point[1]-0.2)
        plotNode(text, end_point, start_point, leafNode, ax)
        old_end_point = end_point

    return layer, old_end_point


def load_model_param(model):
    layer_index_dic = {}
    for index, layer in enumerate(model.layers):
        layer_index_dic[index] = layer

    return layer_index_dic


def visualize_model(model, foldername, pruned_ratio=None):
    ysize = len(model.layers)*90/181
    y_size2 = len(model.layers)*-35/181
    layer_index_dic = load_model_param(model)
    fig, ax = plt.subplots(figsize=(20, ysize))
    ax.set(xlim=(-5, 5), ylim=(y_size2, 3))
    distance = 3
    old_end_point = (0.5, 3)
    layer = model.layers[1]

    while len(layer.outbound_nodes) != 0:
        layer, old_end_point = plot_sequence(
            layer, old_end_point, ax, pruned_ratio, layer_index_dic)

        while len(layer.outbound_nodes) == 2:
            left_start_point = (old_end_point[0]-distance, old_end_point[1])
            right_start_point = (old_end_point[0]+distance, old_end_point[1])
            _, old_end_point = plot_sequence(
                layer.outbound_nodes[0].layer, left_start_point, ax, pruned_ratio, layer_index_dic)
            layer, _ = plot_sequence(
                layer.outbound_nodes[1].layer, right_start_point, ax, pruned_ratio, layer_index_dic)
            old_end_point = (old_end_point[0] + distance, old_end_point[1])

            text = f'{layer.name}'
            start_point = (old_end_point[0], old_end_point[1]-0.05)
            end_point = (old_end_point[0], old_end_point[1]-0.2)
            plotNode(text, end_point, start_point, leafNode, ax)
            old_end_point = end_point

        if len(layer.outbound_nodes) != 0:
            layer = layer.outbound_nodes[0].layer
    plt.savefig(foldername+"/model_plot.png")
    plt.clf()


def plot_model_decompose(model, foldername):
    dot_img_file = foldername+"/decomposed_plot.png"
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

def model_cmp_flops_plot(original_model, compressed_model, foldername):
    result_original = net_flops(original_model)
    result_compressed = net_flops(compressed_model)
    parameter_list = [result_original, result_compressed]
    names = ['Original Model', 'Compressed model']
    r = [0, 1]
    big_k = []
    small_k = []
    depth = []
    rect = []
    others = []
    for l in parameter_list:
        big_k.append(l[0])
        small_k.append(l[1])
        depth.append(l[2])
        rect.append(l[3])
        others.append(l[-1]-l[0]-l[2]-l[1]-l[3])

    barWidth = 0.5
    plt.bar(r, others, color='tab:purple',  width=barWidth)
    plt.bar(r, depth, bottom=others, color='tab:red',  width=barWidth)
    plt.bar(r, rect, bottom=[depth[i]+others[i] for i in range(len(depth))], color='tab:green',  width=barWidth)
    plt.bar(r, small_k, bottom=[depth[i]+others[i]+rect[i] for i in range(len(depth))], color='tab:orange',  width=barWidth)
    plt.bar(r, big_k, bottom=[depth[i]+others[i]+rect[i]+small_k[i] for i in range(len(depth))], color='tab:blue',  width=barWidth)

    plt.xticks(r, names, fontweight='bold')
    plt.xlabel("group")

    # Show graphic
    colors = {
        '2D Convolution with large kernel': 'tab:blue',
        '2D Convolution with small kernel': 'tab:orange',
        'Rectangular 2D Convolution': 'tab:green',
        'Depthwise 2D Convolution': 'tab:red',
        'Others': 'tab:purple'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.ylabel("FLOPs")
    plt.grid()
    plt.savefig(foldername+"/FLOPs_comparison.png")
    plt.clf()
    # plt.show()