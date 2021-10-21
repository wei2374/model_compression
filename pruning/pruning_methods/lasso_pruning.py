from dataset.load_datasets import get_data_from_dataset
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from sklearn.linear_model import LassoLars
from timeit import default_timer as timer
from sklearn.linear_model import LinearRegression
from pruning.helper_functions import rel_error, get_layer_index, load_model_param
from pruning.delete_filters import delete_filter_before
from tools.progress.bar import Bar


def extract_inputs_and_outputs(
                model,
                layer,
                layer_index_dic,
                dataset="food20",
                batches_n=80,
                activation=False):
    index = get_layer_index(layer_index_dic, layer)
    if layer.use_bias:
        bias = layer.get_weights()[1]

    [_, H, W, C] = layer.output_shape
    [h, w] = layer.kernel_size
    fore_layer = layer.inbound_nodes[0].inbound_layers
    try:
        fore_layer_index = get_layer_index(layer_index_dic, fore_layer)
    except Exception:
        fore_layer_index = get_layer_index(layer_index_dic, fore_layer[0])
    train_data, _ = get_data_from_dataset(dataset, batch_size=4)
    inputs = []
    outputs = []

    if activation:
        get_layer_input = K.function([model.layers[0].input],
                                     [model.layers[fore_layer_index].output])
        get_layer_output = K.function([model.layers[0].input],
                                      [model.layers[index+1].output])

    else:
        get_layer_input = K.function([model.layers[0].input],
                                     [model.layers[fore_layer_index].output])
        get_layer_output = K.function([model.layers[0].input],
                                      [model.layers[index].output])

    for batch in range(batches_n):
        it = iter(train_data)
        batch = next(train_data)
        layer_input = get_layer_input([batch[0]])[0]
        layer_output = get_layer_output([batch[0]])[0]

        if activation:
            X = []
            Y = layer_output.reshape((-1, layer_output.shape[3]))
            outputs.append(np.vstack(Y))
            inputs = outputs

        else:
            hh = (h-1)/2
            hw = (w-1)/2
            x_samples = np.random.randint(1, H - 3, 10)
            y_samples = np.random.randint(1, W - 3, 10)
            if layer.use_bias:
                for b in layer_output:
                    for l1 in range(b.shape[2]):
                        b[:, :, l1] = b[:, :, l1]-bias[l1]

            Xs = []
            Ys = []
            for n, x in enumerate(x_samples):
                Y = layer_output[:, x, y_samples[n], :]
                x = x*layer.strides[0]
                y_samples[n] = y_samples[n]*layer.strides[1]
                X = layer_input[
                    :, int(x-hh):int(x+hh+1), int(y_samples[n]-hw):int(y_samples[n]+hw+1), :]
                Xs.append(X)
                Ys.append(Y)
            inputs.append(np.stack(Xs))
            outputs.append(np.vstack(Ys))

    return [np.vstack(np.vstack(inputs)), np.vstack(outputs)]


def featuremap_reconstruction(x, y, copy_x=True, fit_intercept=False):
    """Given changed input X, used linear regression to reconstruct original Y

        Args:
        x: The pruned input
        y: The original feature map of the convolution layer
        Return:
        new weights and bias which can reconstruct the feature map with small loss given X
    """
    _reg = LinearRegression(n_jobs=-1, copy_X=copy_x, fit_intercept=fit_intercept)
    _reg.fit(x, y)
    return _reg.coef_, _reg.intercept_


def compute_pruned_kernel(
        X,
        W2,
        Y,
        alpha=1e-4,
        c_new=None,
        idx=None,
        tolerance=0.02):
    """compute which channels to be pruned by lasso"""
    nb_samples = X.shape[0]
    c_in = X.shape[-1]
    c_out = W2.shape[-1]
    samples = np.random.randint(0, nb_samples, min(400, nb_samples // 20))
    reshape_X = np.rollaxis(
      np.transpose(X, (0, 3, 1, 2)).reshape((nb_samples, c_in, -1))[samples], 1, 0)
    reshape_W2 = np.transpose(np.transpose(W2, (3, 2, 0, 1)).reshape((c_out, c_in, -1)), [1, 2, 0])
    product = np.matmul(reshape_X, reshape_W2).reshape((c_in, -1)).T
    reshape_Y = Y[samples].reshape(-1)

    solver = LassoLars(alpha=alpha, fit_intercept=False, max_iter=3000)

    def solve(alpha):
        """ Solve the Lasso"""
        solver.alpha = alpha
        solver.fit(product, reshape_Y)
        idxs = solver.coef_ != 0.
        tmp = sum(idxs)
        return idxs, tmp, solver.coef_

    print("pruned channel selecting")
    start = timer()

    if c_new == c_in:
        idxs = np.array([True] * c_new)
        # newW2 = W2.reshape(W2.shape[-1],)
    else:
        left = 0
        right = alpha
        lbound = c_new - tolerance * c_in / 2
        rbound = c_new + tolerance * c_in / 2
        while True:
            _, tmp, coef = solve(right)
            if tmp < c_new:
                break
            else:
                right *= 2
            print("relax right to {}".format(right))

        while True:
            if lbound < 0:
                lbound = 1
            idxs, tmp, coef = solve(alpha)
            # print loss
            loss = 1 / (2 * float(product.shape[0])) * np.sqrt(
                np.sum((reshape_Y - np.matmul(
                    product, coef)) ** 2, axis=0)) + alpha * np.sum(np.fabs(coef))

            if lbound <= tmp and tmp <= rbound:
                if False:
                    if tmp % 4 == 0:
                        break
                    elif tmp % 4 <= 2:
                        rbound = tmp - 1
                        lbound = lbound - 2
                    else:
                        lbound = tmp + 1
                        rbound = rbound + 2
                else:
                    break
            elif abs(left - right) <= right * 0.1:
                if lbound > 1:
                    lbound = lbound - 1
                if rbound < c_in:
                    rbound = rbound + 1
                left = left / 1.2
                right = right * 1.2
            elif tmp > rbound:
                left = left + (alpha - left) / 2
            else:
                right = right - (right - alpha) / 2

            if alpha < 1e-10:
                break

            alpha = (left + right) / 2
        c_new = tmp

    newW2, _ = featuremap_reconstruction(
        X[:, :, :, idxs].reshape((nb_samples, -1)), Y, fit_intercept=False)
    return idxs, newW2


def prune_kernel_lasso(
                model,
                index,
                layer_params,
                prune_ratio,
                layer_types,
                layer_bias,
                layer_output_shape,
                filters,
                layer_index_dic,
                cp_lasso=True,
                dataset="food20"):
    if prune_ratio < 1:
        left_edge_flag = False
        after_add = False
        layer_index = index
        current_layer = layer_index_dic[layer_index]
        fore_layer = current_layer.inbound_nodes[0].inbound_layers
        while((not fore_layer == []
              and not isinstance(fore_layer, tf.keras.layers.Conv2D)
              and not isinstance(fore_layer, tf.keras.layers.Add)
              or isinstance(fore_layer, tf.keras.layers.DepthwiseConv2D))
              and not len(fore_layer.outbound_nodes) == 2):
            # TODO:: Batch normalization
            fore_layer = fore_layer.inbound_nodes[0].inbound_layers

        if fore_layer == []:
            new_model_param = layer_params
            num_new_filter = layer_params[index][0].shape[-1]
            print("No pruning implemented for start conv layers")
            return new_model_param, num_new_filter, layer_output_shape, filters

        if isinstance(fore_layer, tf.keras.layers.Add):
            after_add = True

        if len(fore_layer.outbound_nodes) == 2:
            print("This conv2D is at the beginning edge")
            next_layer = current_layer.outbound_nodes[0].layer
            while(not isinstance(next_layer, tf.compat.v1.keras.layers.Conv2D)
                  and not isinstance(next_layer, tf.keras.layers.Add)):
                next_layer = next_layer.outbound_nodes[0].layer

            if isinstance(next_layer, tf.compat.v1.keras.layers.Conv2D):
                print("left edge")
                left_edge_flag = True

        ############################################
        if not left_edge_flag and not after_add:
            layer = model.layers[index]
            W = layer_params[index][0]
            [inputs, outputs] = extract_inputs_and_outputs(
                model, layer, layer_index_dic, dataset=dataset)
            error1 = rel_error(
                inputs.reshape(inputs.shape[0], -1).dot(W.reshape(-1, W.shape[-1])), outputs)
            print('feature map rmse: {}'.format(error1))

            error2 = 1
            # while(error2 > 0.05 and prune_ratio < 1):
            nb_channel_new = int((1-prune_ratio)*(layer.input_shape[3]))
            if cp_lasso is True:
                idxs, newW2 = compute_pruned_kernel(inputs, W, outputs, c_new=nb_channel_new)
            else:
                idxs = np.argsort(-np.abs(W).sum((0, 1, 3)))
                mask = np.zeros(len(idxs), bool)
                idxs = idxs[:nb_channel_new]
                mask[idxs] = True
                idxsz = mask
                reg = LinearRegression(fit_intercept=False)
                reg.fit(inputs[:, :, :, idxs].reshape(inputs.shape[0], -1), outputs)
                newW2 = reg.coef_

            error2 = rel_error(
                inputs[:, :, :, idxs].reshape(inputs.shape[0], -1).dot(newW2.T), outputs)
            print('feature map rmse: {}'.format(error2))
            print('prune_ratio is: {}'.format(prune_ratio))
            #    prune_ratio += 0.1
            '''
            if error2 > 0.1 or prune_ratio > 0.9:
                print("BIG ERROR")
                print('Prune {} c_in from {} to {}'.format(layer.name, inputs.shape[-1], sum(idxs)))
                new_model_param = layer_params
                num_new_filter = layer_params[index][0].shape[-1]
                print("No pruning implemented for left edge conv layers")

            else:
            '''
            print("PRUN IT")
            print('Prune {} c_in from {} to {}'.format(layer.name, inputs.shape[-1], sum(idxs)))
            prun_filter = []
            for i, idx in enumerate(idxs):
                if not idx:
                    prun_filter.append(i)

            filters[index] = prun_filter
            num_new_filter = W.shape[-1]-len(prun_filter)

            h, w = layer.kernel_size
            newW2 = newW2.reshape(-1, h, w, np.sum(idxs))
            newW2 = np.transpose(newW2, [1, 2, 3, 0])
            layer_params[index][0] = newW2
            prun_filter = [prun_filter]
            for i in range(len(prun_filter)-1, -1, -1):
                new_model_param, layer_output_shape = delete_filter_before(
                    layer_params, layer_types, layer_output_shape, layer_bias,
                    index, prun_filter[i], layer_index_dic)

        else:
            new_model_param = layer_params
            num_new_filter = layer_params[index][0].shape[-1]
            print("No pruning implemented for left edge conv layers")

    else:
        new_model_param = layer_params
        num_new_filter = layer_params[index][0].shape[-1]
        print("No pruning implemented for conv layers")

    return new_model_param, num_new_filter, layer_output_shape, filters


def channel_prune_model_lasso(
            my_model,
            prune_ratio,
            min_index=3,
            max_index=None,
            dataset="food20"):
    layer_types, layer_params, layer_output_shape, layer_bias, layer_index_dic = load_model_param(
        my_model)
    max_index = len(my_model.layers) if max_index is None else max_index
    counter = 0
    filters = {}
    with Bar(f'Lasso channel pruning...') as bar:
        for index, layer in enumerate(my_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) and\
                    not isinstance(layer, tf.keras.layers.DepthwiseConv2D) and\
                    layer.kernel_size[0] >= 1 and\
                    index >= min_index and index <= max_index:
                if index >= min_index:
                    layer_params, _, layer_output_shape, filters = prune_kernel_lasso(
                                                        my_model,
                                                        index,
                                                        layer_params,
                                                        prune_ratio[index],
                                                        layer_types,
                                                        layer_bias,
                                                        layer_output_shape,
                                                        filters=filters,
                                                        layer_index_dic=layer_index_dic,
                                                        dataset=dataset)
                    counter += 1
                else:
                    layer_params, _, layer_output_shape, filters = prune_kernel_lasso(
                                                        my_model,
                                                        index,
                                                        layer_params,
                                                        1.0,
                                                        layer_types,
                                                        layer_bias,
                                                        layer_output_shape,
                                                        filters=filters,
                                                        layer_index_dic=layer_index_dic,
                                                        dataset=dataset)
    return layer_params, layer_types
