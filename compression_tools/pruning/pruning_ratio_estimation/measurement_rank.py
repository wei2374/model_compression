from collections import defaultdict, namedtuple
from tensorflow.python.keras.utils.generic_utils import default
from compression_tools.decompose.model_modifier import model_decompose
from sparsity_estimation.rank_estimator import estimate_ranks_VBMF, estimate_ranks_param
import numpy as np
import tensorflow as tf
from model_training.training import validate_model
from scipy.interpolate import PchipInterpolator
from matplotlib import pyplot as plt


def checkIfDuplicates_2(listOfElems):
    ''' Check if given list contains any duplicates '''    
    setOfElems = set()
    for id, elem in enumerate(listOfElems):
        if elem in setOfElems:
            return id
        else:
            setOfElems.add(elem)


def get_estimation(ranks, acc_drop, flops_drop, channels):
    id = checkIfDuplicates_2(ranks)
    if id is not None:
        ranks.remove(ranks[id])
        acc_drop.remove(acc_drop[id])
        flops_drop.remove(flops_drop[id])

    first_acc = np.min(acc_drop)
    first_id = np.argmin(acc_drop)
    for id, acc in enumerate(acc_drop):
        if id < first_id:
            acc_drop[id] = first_acc

    last_acc = np.max(acc_drop)
    last_id = np.argmax(acc_drop)
    for id, acc in enumerate(acc_drop):
        if id > last_id:
            acc_drop[id] = last_acc

    acc_func = PchipInterpolator(ranks, np.asarray(acc_drop))
    flops_func = PchipInterpolator(ranks, np.asarray(flops_drop))
    acc_predict = [acc_func(r+1) for r in range(channels)]
    flops_predict = [flops_func(r+1) for r in range(channels)]

    # first_acc = acc_drop[0]
    # first_id = ranks[0]
    # for id, acc in enumerate(acc_predict):
    #     if id < first_id:
    #         acc_predict[id] = first_acc

    # min_id = ranks[-1] #np.argmin(acc_drop)
    # min_acc = np.min(acc_drop)
    # for id, acc in enumerate(acc_predict):
    #     if id < min_id:
    #         acc_predict[id] = min_acc

    max_id = np.argmax(acc_predict)
    max_acc = np.max(acc_predict)
    for id, acc in enumerate(acc_predict):
        if id > max_id:
            acc_predict[id] = max_acc
    return acc_predict, flops_predict


def single_layer_estimate(model, acc_th):
    x = np.load("res_out_ranks.npy", allow_pickle=True).item()
    input_acc = np.load("res_in_acc.npy", allow_pickle=True).item()
    input_flops = np.load("res_in_flops.npy", allow_pickle=True).item()
    output_acc = np.load("res_out_acc.npy", allow_pickle=True).item()
    output_flops = np.load("res_in_flops.npy", allow_pickle=True).item()
    input_ranks = np.load("res_out_ranks.npy", allow_pickle=True).item()
    output_ranks = np.load("res_out_ranks.npy", allow_pickle=True).item()

    # x = np.load("rank_res_t.npy", allow_pickle=True).item()
    # accuracy_drop = np.load("acc_res_t.npy", allow_pickle=True).item()
    # flops_reduction = np.load("flops_res_t.npy", allow_pickle=True).item()
    # input_ranks = {}
    # output_ranks = {}
    # input_acc_drop = {}
    # output_acc_drop = {}
    # input_flops_drop = {}
    # output_flops_drop = {}
    # for id in x:
    #     input_ranks[id] = [x[id][i] for i in range(1, len(x[id]), 2)]
    #     output_ranks[id] = [x[id][i] for i in range(0, len(x[id]), 2)]
    #     input_acc_drop[id] = [accuracy_drop[id][i] for i in range(1, len(accuracy_drop[id]), 2)]
    #     output_acc_drop[id] = [accuracy_drop[id][i] for i in range(0, len(accuracy_drop[id]), 2)]
    #     input_flops_drop[id] = [flops_reduction[id][i] for i in range(1, len(flops_reduction[id]), 2)]
    #     output_flops_drop[id] = [flops_reduction[id][i] for i in range(0, len(flops_reduction[id]), 2)]

    input_acc_predict = {}
    input_flops_predict = {}
    output_acc_predict = {}
    output_flops_predict = {}

    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.kernel_size[0] > 1 and index > 2:
            # input_acc_predict[index], input_flops_predict[index] = get_estimation(
            #                             output_ranks[index],
            #                             input_acc[index],
            #                             input_flops[index],
            #                             layer.input_shape[3])
            output_acc_predict[index], output_flops_predict[index] = get_estimation(
                                        input_ranks[index],
                                        output_acc[index],
                                        output_flops[index],
                                        layer.output_shape[3])

    # acc_th = 0.75
    if acc_th != 1:
        ranks_result = defaultdict(list)
        for index, la in enumerate(output_acc_predict):
            for i, acc in enumerate(output_acc_predict[la]):
                if acc > acc_th:
                    ranks_result[la].append(i)
                    break
    else:
        ranks_result = defaultdict(list)
        for index, la in enumerate(output_acc_predict):
            ranks_result[la].append(input_ranks[la][-1])

    plt.plot(output_acc_predict[8])
    plt.xlabel("ranks")
    plt.ylabel("validation_accuracy")
    plt.title("accuracy versus rank interpolation for 6th layer of VGG16")
    plt.show()
    # for index, la in enumerate(input_acc_predict):
    #     for i, acc in enumerate(input_acc_predict[la]):
    #         if acc <= acc_drop:
    #             ranks_result[la].append(i)
    #             break

            # 0.7472
    return ranks_result



def multi_layer_estimate(model, high_limit_c, low_limit_c):
    x = np.load("rank.npy", allow_pickle=True).item()
    accuracy_drop = np.load("acc.npy", allow_pickle=True).item()
    flops_reduction = np.load("flops.npy", allow_pickle=True).item()
    acc_predict = {}
    flops_predict = {}
    ranks_out = {}

    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.kernel_size[0] > 1:
            acc_func = PchipInterpolator(x[index], np.asarray(accuracy_drop[index]))
            flops_func = PchipInterpolator(x[index], np.asarray(flops_reduction[index]))
            acc_predict[index] = [acc_func(r+1) for r in range(layer.filters)]
            flops_predict[index] = [flops_func(r+1) for r in range(layer.filters)]
            first_acc = accuracy_drop[index][0]
            first_id = x[index][0]
            for id, acc in enumerate(acc_predict[index]):
                if id < first_id:
                    acc_predict[index][id] = first_acc

            min_id = np.argmin(acc_predict[index])
            min_acc = np.min(acc_predict[index])
            for id, acc in enumerate(acc_predict[index]):
                if id > min_id:
                    acc_predict[index][id] = min_acc

            max_id = np.argmax(acc_predict[index])
            max_acc = np.max(acc_predict[index])
            for id, acc in enumerate(acc_predict[index]):
                if id < max_id:
                    acc_predict[index][id] = max_acc

    rank_space = namedtuple('rank_space', 'index, low_limit, high_limit')
    searching_space = defaultdict(rank_space)
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.kernel_size[0] > 1:
            low_limit_r, _ = estimate_ranks_param(layer, low_limit_c)
            high_limit_r, _ = estimate_ranks_param(layer, high_limit_c)
            searching_space[index] = rank_space(index, low_limit_r, high_limit_r)

    return ranks_out


def estimate_ranks_measurement(model, acc):
    ranks_out = single_layer_estimate(model, acc)
    return ranks_out
    his, flops_ori, _ = validate_model(model=model, dataset="food20", split=0.05)
    # target_C = int(flops_ori*ratio_complexity)
    original_acc = his[0]
    ranks_out = {}
    acc_predict = {}
    flops_predict = {}
    est_rank = {}
    in_ranks = {}
    out_ranks = {}
    in_acc = {}
    out_acc = {}
    in_flops = {}
    out_flops = {}
    for index, layer in enumerate(model.layers):
        # layer = model.layers[29]
        ok_to_leave = False
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.kernel_size[0] > 1 and index>2:
            # r = estimate_ranks_VBMF(model.layers[1], method="channel", parameter=0.003)
            input_r = []
            output_r = []
            input_accuracy = []
            output_accuracy = []
            input_flops = []
            output_flops = []
            up_limit = 0.005
            samples = 10
            step = up_limit/samples

            # while not ok_to_leave:
            for s in range(samples-1):
                param = up_limit-(s+1)*step
                r = estimate_ranks_VBMF(layer, parameter=param)
                if r[0] == 0:
                    r[0] = 1
                if r[0] in output_r:
                    continue
                # if r[1] == 0:
                #     r[1] = 1
                # if r[1] in input_r:
                #     continue

                modified_model = model_decompose(
                    model,
                    method="channel_output",
                    ranks=r[0],
                    rank_selection=None,
                    min_index=index,
                    max_index=index,
                    big_kernel_only="True",
                    option="CL"
                    )
                his, flops, params = validate_model(
                                model=modified_model,
                                dataset="food20",
                                split=0.1)
                output_accuracy.append(his[0])
                output_flops.append(flops)
                output_r.append(r[0])

                # modified_model = model_decompose(
                #     model,
                #     method="channel_input",
                #     ranks=r[1],
                #     rank_selection=None,
                #     min_index=index,
                #     max_index=index,
                #     big_kernel_only="True",
                #     option="CL"
                #     )
                # his, flops, params = validate_model(
                #                 model=modified_model,
                #                 dataset="food20",
                #                 split=0.2)

                input_accuracy.append(his[0])
                input_flops.append(flops)
                input_r.append(r[1])
            in_ranks[index] = input_r
            out_ranks[index] = output_r
            in_acc[index] = input_accuracy
            out_acc[index] = output_accuracy
            in_flops[index] = input_flops
            out_flops[index] = output_flops

    np.save("res_in_ranks.npy", in_ranks)
    np.save("res_out_ranks.npy", out_ranks)
    np.save("res_in_acc.npy", in_acc)
    np.save("res_out_acc.npy", out_acc)
    np.save("res_in_flops.npy", in_flops)
    np.save("res_out_flops.npy", out_flops)
