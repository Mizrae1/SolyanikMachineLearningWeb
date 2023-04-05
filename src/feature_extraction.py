import numpy as np


def acc_diff_start_end_values(list_data):
    """
    function to compute differences between the final value of the function and the initial one
    :param list_data: list - [[train_data], [test_data]] for accuracy values
    :return: list of two values of the differences
    """
    return [list_data[0][len(list_data) - 1] - list_data[0][0], list_data[1][len(list_data) - 1] - list_data[1][0]]


def acc_loss_standard_deviation_diff(list_data):
    """
    function to compute standard deviation of differences f1(x) - f2(x) for all x
    :param list_data: list - [[train_data], [test_data]]
    :return: list of standard deviation
    """
    diff = []
    for train_data, test_data in zip(list_data[0], list_data[1]):
        diff.append(train_data - test_data)
    return [np.std(diff)]


def acc_loss_standard_deviation(list_data):
    """
    function to compute standard deviation f1(x) and f2(x) for all x
    :param list_data: list - [[train_data], [test_data]]
    :return: list of standard deviations
    """
    return [np.std(list_data[0]), np.std(list_data[1])]


def acc_diff_end_values(list_data):
    """
    function to compute difference between the end value of the f1 and the end value of the f2
    :param list_data: list - [[train_data], [test_data]] for accuracy values
    :return: list of value of the difference
    """
    return [list_data[0][len(list_data) - 1] - list_data[1][len(list_data) - 1]]


def acc_area_under_curves(list_data):
    """
    function to compute area under the curves f1 and f2
    :param list_data: list - [[train_data], [test_data]] for accuracy values
    :return: list of two values of the area
    """
    return [np.trapz(list_data[0]), np.trapz(list_data[1])]


def acc_max_values(list_data):
    """
    function to compute area under the curves f1 and f2
    :param list_data: list - [[train_data], [test_data]] for accuracy values
    :return: list of two values of the area
    """
    return [max(list_data[0]), max(list_data[1])]


def acc_loss_mean_gradient(list_data):
    """
    function to compute mean gradients of the f1 and f2
    :param list_data: list - [[train_data], [test_data]] for accuracy values
    :return: list of two values of mean gradients
    """
    eps = 5
    return [np.mean(np.gradient(list_data[0][-eps:])), np.mean(np.gradient(list_data[1][-eps:]))]


def acc_loss_std_last_epochs(list_data):
    """
    function to compute standard deviation f1(x) and f2(x) for all x on 20% last epochs
    :param list_data: list - [[train_data], [test_data]]
    :return: list of standard deviations
    """
    num_epochs = int(len(list_data[0]) * 0.2)
    return [np.std(list_data[0][-num_epochs:]), np.std(list_data[1][-num_epochs:])]


def acc_diff_start_end_values_last_epochs(list_data):
    """
    function to compute differences between the final value of the function and the initial one on 20% last epochs
    :param list_data: list - [[train_data], [test_data]] for accuracy values
    :return: list of two values of the differences
    """
    num_epochs = int(len(list_data[0]) * 0.2)
    return [list_data[0][len(list_data) - 1] - list_data[0][-num_epochs],
            list_data[1][len(list_data) - 1] - list_data[1][-num_epochs]]


def acc_all_features(list_data):
    """
    function to compute all features for accuracy curves
    :param list_data: list - [[train_data], [test_data]] for accuracy values
    :return: list of features
    """
    return [*acc_diff_start_end_values(list_data), *acc_loss_standard_deviation_diff(list_data),
            *acc_loss_standard_deviation(list_data), *acc_diff_end_values(list_data), *acc_area_under_curves(list_data),
            *acc_max_values(list_data), *acc_loss_mean_gradient(list_data), *acc_loss_std_last_epochs(list_data),
            *acc_diff_start_end_values_last_epochs(list_data)]


def loss_all_features(list_data):
    """
    function to compute all features for loss curves
    :param list_data: list - [[train_data], [test_data]] for accuracy values
    :return: list of features
    """
    return [*acc_loss_standard_deviation_diff(list_data), *acc_loss_standard_deviation(list_data),
            *acc_loss_mean_gradient(list_data), *acc_loss_std_last_epochs(list_data)]


def acc_feature_names():
    return ['diff_start_end_values_train', 'diff_start_end_values_test', 'std_diff_functions', 'std_train',
            'std_test', 'diff_end_values', 'area_train', 'area_test', 'max_value_train', 'max_value_test',
            'mean_gradient_train', 'mean_gradient_test', 'std_last_20%_epochs_train', 'std_last_20%_epochs_test',
            'diff_start_end_values_last_20%_epochs_train', 'diff_start_end_values_last_20%_epochs_test', 'F1_train',
            'F2_train', 'F3_train', 'F1_test', 'F2_test', 'F3_test', 'target']


def loss_feature_names():
    return ['std_diff_functions', 'std_train', 'std_test', 'mean_gradient_train',
            'mean_gradient_test', 'std_last_20%_epochs_train', 'std_last_20%_epochs_test',
            'F1_train', 'F2_train', 'F3_train', 'F1_test', 'F2_test', 'F3_test', 'target']


def f_features(list_data):
    """
    function to compute f1, f2 adn f3 features
    :param list_data: list - [[train_data], [test_data]]
    :return: list of six values of features
    """
    step = len(list_data[0]) // 4
    for i in range(step):
        F1_train = - list_data[0][i] - list_data[0][i + step] + list_data[0][i + step * 2] + list_data[0][i + step * 3]
        F2_train = - list_data[0][i] + list_data[0][i + step] + list_data[0][i + step * 2] - list_data[0][i + step * 3]
        F3_train = list_data[0][i] - list_data[0][i + step] + list_data[0][i + step * 2] - list_data[0][i + step * 3]
        F1_test = - list_data[1][i] - list_data[1][i + step] + list_data[1][i + step * 2] + list_data[1][i + step * 3]
        F2_test = - list_data[1][i] + list_data[1][i + step] + list_data[1][i + step * 2] - list_data[1][i + step * 3]
        F3_test = list_data[1][i] - list_data[1][i + step] + list_data[1][i + step * 2] - list_data[1][i + step * 3]
        return [F1_train, F2_train, F3_train, F1_test, F2_test, F3_test]
