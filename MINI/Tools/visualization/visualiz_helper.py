from keras_flops import flops_calculation
import matplotlib.pyplot as plt
import numpy as np
import collections


def compare_sparsity_FT():
    flops_acc_ori = np.load("result_energy.npy", allow_pickle=True).item()
    flops_acc_ft = np.load("result_energy_ft.npy", allow_pickle=True).item()

    x_axis = []
    y_axis = []
    for param in flops_acc_ori:
        x_axis.append(flops_acc_ori[param][1])
        y_axis.append(flops_acc_ori[param][0])

    plt.plot(x_axis, y_axis, color="blue", label="validation")

    x_axis = []
    y_axis = []
    for param in flops_acc_ft:
        x_axis.append(flops_acc_ft[param][1])
        y_axis.append(flops_acc_ft[param][0])

    plt.plot(x_axis, y_axis, color="red", label="fine_tuning")

    x_axis = []
    y_axis = []

    plt.xlabel('flops')
    plt.ylabel('validation accuracy')
    plt.title(f"Rank estimation comparison")
    plt.legend()
    plt.show()


def compare_decomposition_schema(type):
    if type == "vgg16":
        model_dir = "/home/wei-bshg/Documents/params/vgg16_decomposition_strategies/"
        original_point = [30800000000, 0.7781]
        channel_all_vgg16_flops_acc_param = np.load(model_dir+"new_vgg16_channel_all_Param_ft.npy", allow_pickle=True).item()
        channel_out_vgg16_flops_acc_param = np.load(model_dir+"result_VGG16_channel_out_Param_ft.npy", allow_pickle=True).item()
        VH_vgg16_flops_acc_param = np.load(model_dir+"new_vgg16_VH_Param_ft.npy", allow_pickle=True).item()
        depth_vgg16_flops_acc_param = np.load(model_dir+"new_vgg16_depthwise_energy_ft.npy", allow_pickle=True).item()
        tucker2D_vgg16_flops_acc_param = np.load(model_dir+"result_VGG16_tucker2D_Param_ft.npy", allow_pickle=True).item()
        CP_flops_acc_param = np.load(model_dir+"result_VGG16_CP_Param_ft.npy", allow_pickle=True).item()
        CNL_flops_acc_param = np.load(model_dir+"result_VGG16_CNL_Param_ft.npy", allow_pickle=True).item()
    else:
        model_dir = "/home/wei-bshg/Documents/params/res50_decomposition_strategies/"
        original_point = [7800000000, 0.8727]
        channel_all_vgg16_flops_acc_param = np.load(model_dir+"new_res50_channel_all_Param_ft.npy", allow_pickle=True).item()
        channel_out_vgg16_flops_acc_param = np.load(model_dir+"new_res50_channel_output_Param_ft.npy", allow_pickle=True).item()
        VH_vgg16_flops_acc_param = np.load(model_dir+"new_res50_VH_Param_ft.npy", allow_pickle=True).item()
        depth_vgg16_flops_acc_param = np.load(model_dir+"new_res50_Param_depthwise_ft.npy", allow_pickle=True).item()
        tucker2D_vgg16_flops_acc_param = np.load(model_dir+"new_res50_tucker2D_param_ft.npy", allow_pickle=True).item()
        CP_flops_acc_param = np.load(model_dir+"result_resnet50_CP_Param_ft.npy", allow_pickle=True).item()
        CNL_flops_acc_param = np.load(model_dir+"result_resnet50_channel_nl2_Param_ft.npy", allow_pickle=True).item()

    x_axis = []
    y_axis = []
    id = 0
    for param in channel_all_vgg16_flops_acc_param:
        x_axis.append(channel_all_vgg16_flops_acc_param[param][-1])
        y_axis.append(channel_all_vgg16_flops_acc_param[param][id])
    plt.plot(x_axis, y_axis, color="blue", label="channel_all")

    x_axis = []
    y_axis = []
    for param in channel_out_vgg16_flops_acc_param:
        x_axis.append(channel_out_vgg16_flops_acc_param[param][-1])
        y_axis.append(channel_out_vgg16_flops_acc_param[param][id])
    plt.plot(x_axis, y_axis, color="red", label="channel_out")

    x_axis = []
    y_axis = []
    for param in VH_vgg16_flops_acc_param:
        x_axis.append(VH_vgg16_flops_acc_param[param][-1])
        y_axis.append(VH_vgg16_flops_acc_param[param][id])
    plt.plot(x_axis, y_axis, color="green", label="VH")

    x_axis = []
    y_axis = []
    for param in depth_vgg16_flops_acc_param:
        x_axis.append(depth_vgg16_flops_acc_param[param][-1])
        y_axis.append(depth_vgg16_flops_acc_param[param][id])
    plt.plot(x_axis, y_axis, color="yellow", label="depthwise")

    x_axis = []
    y_axis = []
    for param in tucker2D_vgg16_flops_acc_param:
        x_axis.append(tucker2D_vgg16_flops_acc_param[param][-1])
        y_axis.append(tucker2D_vgg16_flops_acc_param[param][id])
    plt.plot(x_axis, y_axis, color="purple", label="tucker2D")

    x_axis = []
    y_axis = []
    for param in CP_flops_acc_param:
        x_axis.append(CP_flops_acc_param[param][-1])
        y_axis.append(CP_flops_acc_param[param][0])
    plt.plot(x_axis, y_axis, color="grey", label="CP")

    x_axis = []
    y_axis = []
    for param in CNL_flops_acc_param:
        x_axis.append(CNL_flops_acc_param[param][-1])
        y_axis.append(CNL_flops_acc_param[param][id])
    plt.plot(x_axis, y_axis, color="black", label="CNL")
    plt.plot(original_point[0], original_point[1], 'gx', label="original")
    plt.hlines(original_point[1], np.min(x_axis), original_point[0], linestyles="dashed")

    plt.xlabel('flops')
    plt.ylabel('validation accuracy')
    plt.title(f"Decomposition comparison for {type}")
    plt.legend()
    filename = f"decomposition_comparison_for_{type}.png"
    plt.savefig(filename)
    plt.show()


def compare_sparsity_validation(type="resnet"):
    if type == "resnet50":
        flops_acc_energy = np.load("result_RES_t_energy_ft.npy", allow_pickle=True).item()
        flops_acc_param = np.load("result_RES_t_param_ft.npy", allow_pickle=True).item()
        flops_acc_vbmf = np.load("new_res50_VBMF_ft.npy", allow_pickle=True).item()
        original_accuracy = 0.8795
        original_point = [7800000000, 0.8727]
        vbmf_point = [4908575688, 0.84460, 0.7699]
        # bayes_point = [4787505998, 0.82640, 0.7842]
        bayes_point = [5024763410, 0.86640, 0.8156]
        random_point = [4956273464, 0.77900]
        # Vbmf_point = [4916859824, 0.8304]
        param_point = [4899115454, 0.85660]
        energy_param = [4881725158, 0.85620]

    elif type == "vgg16":
        flops_acc_energy = np.load("result_VGG16_t_energy_ft.npy", allow_pickle=True).item()
        flops_acc_param = np.load("result_VGG16_t_param_ft.npy", allow_pickle=True).item()
        flops_acc_vbmf = np.load("result_VGG16_t_VBMF_ft.npy", allow_pickle=True).item()
        # flops_acc_mea = np.load("result_VGG16_c_measurement_ft.npy", allow_pickle=True).item()
        original_accuracy = 0.7810
        original_point = [30800000000, 0.7781]
        vbmf_point = [8229325044, 0.7558, 0.7459]
        # vbmf_point = [13393569892, 0.7560, 0.7594]
        bayes_point = [6191048916, 0.7612, 0.7316]
        random_point = []
        Vbmf_point = [6622362988, 0.75400]
        energy_point = [8120524660, 0.74600]
        param_point = [7588546948, 0.7576]

    x_axis = []
    y_axis = []
    for param in flops_acc_energy:
        x_axis.append(flops_acc_energy[param][2])
        x_axis.sort()
        y_axis.append(flops_acc_energy[param][1])
        y_axis.sort()

    plt.plot(x_axis, y_axis, color="red", label="energy")

    # x_axis = []
    # y_axis = []
    # for param in flops_acc_mea:
    #     x_axis.append(flops_acc_mea[param][2])
    #     y_axis.append(flops_acc_mea[param][1])

    # plt.plot(x_axis, y_axis, color="black", label="measurement")

    x_axis = []
    y_axis = []
    for param in flops_acc_vbmf:
        x_axis.append(flops_acc_vbmf[param][2])
        y_axis.append(flops_acc_vbmf[param][1])

    plt.plot(x_axis, y_axis, color="blue", label="vbmf")

    x_axis = []
    y_axis = []
    for param in flops_acc_param:
        x_axis.append(flops_acc_param[param][2])
        y_axis.append(flops_acc_param[param][1])

    plt.plot(x_axis, y_axis, color="green", label="param")

    plt.plot(vbmf_point[0], vbmf_point[2], 'bx', label="VBMF_auto")
    plt.plot(bayes_point[0], bayes_point[2], 'rx', label="BayesOpt")
    plt.plot(original_point[0], original_point[1], 'gx', label="original")
    plt.hlines(original_accuracy, np.min(x_axis), original_point[0], linestyles="dashed")

    plt.xlabel('flops')
    plt.ylabel('validation accuracy')
    plt.title(f"raw rank estimation comparison for {type}")
    plt.legend()
    # plt.ylim((0.3, 1))

    filename = f"raw_rank_estimation_comparison_for_{type}.png"
    plt.savefig(filename)
    plt.show()


def plot_sparsity_validation(filename, estimation_methods):
    flops_acc = np.load(filename, allow_pickle=True).item()
    x_axis = []
    y_axis = []
    for param in flops_acc:
        x_axis.append(flops_acc[param][1])
        y_axis.append(flops_acc[param][0])

    plt.xlabel('flops')
    plt.ylabel('validation accuracy')
    plt.title(f"Rank estimation with method {estimation_methods}")
    plt.legend()
    plt.show()


def compare_pruning_ratio_estimation(type):
    model_dir = "/home/wei-bshg/Documents/params/vgg16_pruning_strategies/"
    if type == "vgg16":
        original_point = [30800000000, 0.7781]
        uniform_gradient1 = np.load(model_dir+"pruning_vgg16_uniform_g1_ft.npy", allow_pickle=True).item()
        uniform_gradient1 = collections.OrderedDict(sorted(uniform_gradient1.items()))
        whole_gradient1 = np.load(model_dir+"pruning_vgg16_whole_g1_ft.npy", allow_pickle=True).item()
        whole_gradient1 = collections.OrderedDict(sorted(whole_gradient1.items()))
        energy_g1 = np.load(model_dir+"pruning_vgg16_energy_g1_ft.npy", allow_pickle=True).item()
        energy_lasso = np.load(model_dir+"pruning_vgg16_energy_lasso_ft.npy", allow_pickle=True).item()
        lasso = np.load(model_dir+"pruning_VGG16_lasso_ft.npy", allow_pickle=True).item()

    elif type == "resnet50":
        original_point = [7800000000, 0.8727]
        uniform_gradient1 = np.load(model_dir+"pruning_resnet50_Uniform_gradient1_ft.npy", allow_pickle=True).item()
        uniform_gradient1 = collections.OrderedDict(sorted(uniform_gradient1.items()))
        whole_gradient1 = np.load(model_dir+"pruning_res50_whole_g1_ft.npy", allow_pickle=True).item()
        whole_gradient1 = collections.OrderedDict(sorted(whole_gradient1.items()))
        # energy_g1 = np.load(model_dir+"pruning_res50_energy_g1_ft.npy", allow_pickle=True).item()
        # energy_lasso = np.load(model_dir+"pruning_res50_energy_lasso_ft.npy", allow_pickle=True).item()
        lasso = np.load(model_dir+"pruning_resnet50_lasso_ft.npy", allow_pickle=True).item()

    elif type == "mv2":
        original_point = [612977944, 0.7626]
        uniform_gradient1 = np.load(model_dir+"pruning_mv2_uniform_g1_ft.npy", allow_pickle=True).item()
        whole_gradient1 = np.load(model_dir+"pruning_mv2_whole_g1_ft.npy", allow_pickle=True).item()
        energy_g1 = np.load(model_dir+"pruning_mv2_energy_g1_ft.npy", allow_pickle=True).item()
        energy_lasso = np.load(model_dir+"pruning_mv2_energy_lasso_ft.npy", allow_pickle=True).item()
        lasso = np.load(model_dir+"pruning_mv2_lasso_ft.npy", allow_pickle=True).item()


    x_axis = []
    y_axis = []
    if 'uniform_gradient1' in locals():
        for param in uniform_gradient1:
            x_axis.append(uniform_gradient1[param][-1])
            y_axis.append(uniform_gradient1[param][0])
        plt.plot(x_axis, y_axis, color="red", label="uniform")

    x_axis = []
    y_axis = []
    if 'whole_gradient1' in locals():
        for param in whole_gradient1:
            x_axis.append(whole_gradient1[param][-1])
            y_axis.append(whole_gradient1[param][0])
        plt.plot(x_axis, y_axis, color="blue", label="whole")

    if 'energy_g1' in locals():
        x_axis = []
        y_axis = []
        for param in energy_g1:
            x_axis.append(energy_g1[param][-1])
            y_axis.append(energy_g1[param][0])
        plt.plot(x_axis, y_axis, color="green", label="energy_g1")

    if 'energy_lasso' in locals():
        x_axis = []
        y_axis = []
        for param in energy_lasso:
            x_axis.append(energy_lasso[param][-1])
            y_axis.append(energy_lasso[param][0])
        plt.plot(x_axis, y_axis, color="black", label="energy_lasso")

    if 'lasso' in locals():
        x_axis = []
        y_axis = []
        for param in lasso:
            x_axis.append(lasso[param][-1])
            y_axis.append(lasso[param][0])
        plt.plot(x_axis, y_axis, color="purple", label="lasso")

    plt.plot(original_point[0], original_point[1], 'gx', label="original")
    plt.hlines(original_point[1], np.min(x_axis), original_point[0], linestyles="dashed")

    plt.xlabel('flops')
    plt.ylabel('validation accuracy')
    plt.title(f"Pruning criterion comparison for {type}")
    plt.legend()
    filename = f"pruning_ratio_comparison_for_{type}.png"
    plt.savefig(filename)
    plt.show()


def compare_pruning_criterions(type):
    if type == "vgg16":
        original_point = [30800000000, 0.7781]
        uniform_gradient1_resnet = np.load("pruning_VGG16_uniform_gradient1n_ft.npy", allow_pickle=True).item()
        uniform_gradient2_resnet = np.load("pruning_VGG16_Uniform_gradient2_ft.npy", allow_pickle=True).item()
        uniform_magnitude_resnet = np.load("pruning_VGG16_Uniform_magnitude_ft.npy", allow_pickle=True).item()
        uniform_avtivation_resnet = np.load("pruning_VGG16_Uniform_activation_ft.npy", allow_pickle=True).item()
        uniform_ApoZ_resnet = np.load("pruning_VGG16_uniform_ApoZ1n_ft.npy", allow_pickle=True).item()
        lasso_vgg = np.load("pruning_VGG16_lasso_ft.npy", allow_pickle=True).item()
        original_accuracy = 0.7810

    elif type == "mv2":
        original_point = [612977944, 0.7626]
        uniform_gradient1_resnet = np.load("pruning_mv2_Layerwise_g1_ft.npy", allow_pickle=True).item()
        uniform_gradient2_resnet = np.load("pruning_mv2_Layerwise_g2_ft.npy", allow_pickle=True).item()
        uniform_magnitude_resnet = np.load("pruning_mv2_Layerwise_mag_ft.npy", allow_pickle=True).item()
        uniform_avtivation_resnet = np.load("pruning_mv2_Layerwise_act_ft.npy", allow_pickle=True).item()
        uniform_ApoZ_resnet = np.load("pruning_mv2_Layerwise_apoz_ft.npy", allow_pickle=True).item()
        lasso_vgg = np.load("pruning_mv2_Lasso_ft.npy", allow_pickle=True).item()
        original_accuracy = 0.76

    else:
        original_point = [7800000000, 0.8727]
        uniform_gradient1_resnet = np.load("pruning_res50_Layerwise_g1_ft.npy", allow_pickle=True).item()
        uniform_gradient2_resnet = np.load("pruning_res50_Layerwise_g2_ft.npy", allow_pickle=True).item()
        uniform_magnitude_resnet = np.load("pruning_res50_Layerwise_mag_ft.npy", allow_pickle=True).item()
        uniform_avtivation_resnet = np.load("pruning_res50_Layerwise_act_ft.npy", allow_pickle=True).item()
        uniform_ApoZ_resnet = np.load("pruning_res50_Layerwise_apoz_ft.npy", allow_pickle=True).item()
        lasso_vgg = np.load("pruning_res50_Lasso_ft.npy", allow_pickle=True).item()
        original_accuracy = 0.87

    x_axis = []
    y_axis = []
    if 'uniform_gradient1_resnet' in locals():
        for param in uniform_gradient1_resnet:
            x_axis.append(uniform_gradient1_resnet[param][-1])
            y_axis.append(uniform_gradient1_resnet[param][0])
        plt.plot(x_axis, y_axis, color="blue", label="gradient1")

    x_axis = []
    y_axis = []
    if 'uniform_gradient2_resnet' in locals():
        for param in uniform_gradient2_resnet:
            x_axis.append(uniform_gradient2_resnet[param][-1])
            y_axis.append(uniform_gradient2_resnet[param][0])
        plt.plot(x_axis, y_axis, color="red", label="gradient2")

    x_axis = []
    y_axis = []
    if 'uniform_magnitude_resnet' in locals():
        for param in uniform_magnitude_resnet:
            x_axis.append(uniform_magnitude_resnet[param][-1])
            y_axis.append(uniform_magnitude_resnet[param][0])
        plt.plot(x_axis, y_axis, color="green", label="magnitude")

    x_axis = []
    y_axis = []
    if 'lasso_vgg' in locals():
        for param in lasso_vgg:
            x_axis.append(lasso_vgg[param][-1])
            y_axis.append(lasso_vgg[param][0])
        plt.plot(x_axis, y_axis, color="black", label="lasso")

    x_axis = []
    y_axis = []
    if 'uniform_avtivation_resnet' in locals():
        for param in uniform_avtivation_resnet:
            x_axis.append(uniform_avtivation_resnet[param][-1])
            y_axis.append(uniform_avtivation_resnet[param][0])
        plt.plot(x_axis, y_axis, color="yellow", label="activation")

    x_axis = []
    y_axis = []
    if 'uniform_ApoZ_resnet' in locals():
        for param in uniform_ApoZ_resnet:
            x_axis.append(uniform_ApoZ_resnet[param][-1])
            y_axis.append(uniform_ApoZ_resnet[param][0])
        plt.plot(x_axis, y_axis, color="purple", label="ApoZ")

    plt.plot(original_point[0], original_point[1], 'gx', label="original")
    plt.hlines(original_accuracy, np.min(x_axis), original_point[0], linestyles="dashed")

    plt.xlabel('flops')
    plt.ylabel('validation accuracy')
    plt.title(f"Pruning criterion comparison for {type}")
    plt.legend()
    filename = f"pruning_criterions_comparison_for_{type}.png"
    plt.savefig(filename)
    plt.show()


def plot_hardware_inference(type):
    if type == "vgg16":
        original_point = [21, 0.396, 4.63, 0.054, 30723156068, 19827732, 30693261312.0, 0]
        tucker2d_point = [5.32, 0.219, 1.02, 0.031, 7730390972, 8780052]
        # channel_point = [5.56, 0.207, 7748569188, 8784020]
        cp_point = [9.24, 0.279, 2.91, 0.039, 7810442468, 8791830]
        depth_point = [6, 0.307, 1.07, 0.042, 7786602596, 8620372]
        vh_point = [5.19, 0.208, 1.02, 0.029, 7833266276, 8796180]
        g1_point = [5.1, 0.16, 1.05, 0.023, 7731659876, 6259316]
        lasso_point = [5.51, 0.212, 1.10, 0.03, 7974534844, 9376847]

    elif type == "resnet50":
        original_point = [5.4, 0.772, 1.10, 0.118, 7801701112, 48843668, 3948217728, 3776446464]
        tucker2d_point = [3.48, 0.74, 0.67, 0.101, 4940095918, 40115069]
        cp_point = [3.79, 0.675, 0.80, 0.101, 4950419728, 40123466]
        depth_point = [3.61, 0.747, 0.70, 0.10, 4949295864, 39842580]
        vh_point = [3.49, 0.751, 0.70, 0.1, 4950098680, 40119956]
        g1_point = [3.56, 0.704, 0.75, 0.10, 4971348880, 38292184]
        lasso_point = [3.60, 0.704, 0.74, 0.094, 5020569588, 41191724]

    elif type == "mv2":
        original_point = [0.48, 0.077, 0.098, 0.011]
        g1_point = [0.34, 0.063, 0.061, 0.009]
        lasso_point = [0.33, 0.062, 0.062, 0.009]

    x_axis = []
    inference_t = []
    inference_r = []
    mem = []
    color = []
    if 'original_point' in locals():
        x_axis.append("original")
        inference_t.append(original_point[0])
        inference_r.append(original_point[2])
        mem.append(original_point[1])
        color.append('tab:blue')

    if 'tucker2d_point' in locals():
        x_axis.append("Tucker2D")
        inference_t.append(tucker2d_point[0])
        inference_r.append(tucker2d_point[2])
        mem.append(tucker2d_point[1])
        color.append('tab:orange')

    if 'cp_point' in locals():
        x_axis.append("CP")
        inference_t.append(cp_point[0])
        inference_r.append(cp_point[2])
        mem.append(cp_point[1])
        color.append('tab:orange')

    if 'depth_point' in locals():
        x_axis.append("depthwise")
        inference_t.append(depth_point[0])
        inference_r.append(depth_point[2])
        mem.append(depth_point[1])
        color.append('tab:orange')

    if 'vh_point' in locals():
        x_axis.append("VH")
        inference_t.append(vh_point[0])
        inference_r.append(vh_point[2])
        mem.append(vh_point[1])
        color.append('tab:orange')

    if 'g1_point' in locals():
        x_axis.append("g1")
        inference_t.append(g1_point[0])
        inference_r.append(g1_point[2])
        mem.append(g1_point[1])
        color.append('limegreen')

    if 'lasso_point' in locals():
        x_axis.append("lasso")
        inference_t.append(lasso_point[0])
        inference_r.append(lasso_point[2])
        mem.append(lasso_point[1])
        color.append('limegreen')

    X = np.arange(len(x_axis))
    plt.bar(X, inference_t, color=color, width=0.25)
    plt.bar(X+0.35, inference_r, color=color, width=0.25)
    plt.xticks(X+0.175, x_axis)
    plt.xlabel('compression methods')
    plt.ylabel('seconds')
    plt.title(f"Inference time comparison for {type}")
    plt.grid()
    colors = {
        'Original model': 'tab:blue',
        'Decomposed model': 'tab:orange',
        'Pruned model': 'tab:green'
        }
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    filename = f"Inf_time_cmp_for_{type}.png"
    plt.savefig(filename)
    plt.show()


def compare_hardware_inference(type):
    vgg_original = [4.63, 0.054, 2.93, 0.099]
    res_original = [1.10, 0.118, 1.17, 0.097]
    mv_original = [0.098, 0.011, 0.183, 0.015]
    vgg_tucker2d = [1.02, 0.031, 1.42, 0.036]
    vgg_lasso = [1.10, 0.03, 0.76, 0.039]

    x_axis = []
    inference_tflite = []
    inference_tvm = []
    mem_tflite = []
    mem_tvm = []
    color_tflite = []
    color_tvm = []

    if 'vgg_original' in locals():
        x_axis.append("vgg_original")
        inference_tflite.append(vgg_original[0])
        inference_tvm.append(vgg_original[2])
        mem_tflite.append(vgg_original[1])
        mem_tvm.append(vgg_original[1])
        color_tflite.append('tab:blue')
        color_tvm.append('tab:green')

    if 'vgg_tucker2d' in locals():
        x_axis.append("vgg_tucker2d")
        inference_tflite.append(vgg_tucker2d[0])
        inference_tvm.append(vgg_tucker2d[2])
        mem_tflite.append(vgg_tucker2d[1])
        mem_tvm.append(vgg_tucker2d[1])
        color_tflite.append('tab:blue')
        color_tvm.append('tab:green')

    if 'vgg_lasso' in locals():
        x_axis.append("vgg_lasso")
        inference_tflite.append(vgg_lasso[0])
        inference_tvm.append(vgg_lasso[2])
        mem_tflite.append(vgg_lasso[1])
        mem_tvm.append(vgg_lasso[1])
        color_tflite.append('tab:blue')
        color_tvm.append('tab:green')

    if 'res_original' in locals():
        x_axis.append("res_original")
        inference_tflite.append(res_original[0])
        inference_tvm.append(res_original[2])
        mem_tflite.append(res_original[1])
        mem_tvm.append(res_original[1])
        color_tflite.append('tab:blue')
        color_tvm.append('tab:green')

    if 'mv_original' in locals():
        x_axis.append("mv_original")
        inference_tflite.append(mv_original[0])
        inference_tvm.append(mv_original[2])
        mem_tflite.append(mv_original[1])
        mem_tvm.append(mv_original[1])
        color_tflite.append('tab:blue')
        color_tvm.append('tab:green')

    X = np.arange(len(x_axis))
    plt.bar(X, inference_tflite, color=color_tflite, width=0.25)
    plt.bar(X+0.35, inference_tvm, color=color_tvm, width=0.25)
    plt.xticks(X+0.175, x_axis)
    plt.xlabel('compression methods')
    plt.ylabel('seconds')
    plt.title(f"Inference time comparison")
    colors = {
        'TensorFlow Lite': 'tab:blue',
        'TVM auto-tuning': 'tab:green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.grid()
    filename = f"time_cmp.png"
    plt.savefig(filename)
    plt.show()
