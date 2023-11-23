import numpy as np
import scipy.stats as stats


def welch_t_test(mu1, s1, N1, mu2, s2, N2):  # Parametric test, 2 groups, different subjects
    dof1 = N1 - 1
    dof2 = N2 - 1
    v1 = s1 ** 2
    v2 = s2 ** 2

    # Calculate t-stat, degrees of freedom, use scipy to find p-value.
    t = (mu1 - mu2) / np.sqrt(v1 / N1 + v2 / N2)
    dof = (v1 / N1 + v2 / N2) ** 2 / ((v1 / N1) ** 2 / dof1 + (v2 / N2) ** 2 / dof2)
    p = stats.distributions.t.sf(np.abs(t), dof) * 2  # two-sided pvalue

    x = np.where(np.isnan(p))
    p[x] = 1

    return t, p


def paired_t_test(mu_diff, s_diff, N):  # Parametric test, 2 groups, same subjects
    dof = N - 1
    mu_0 = 0

    # Calculate t-stat, degrees of freedom, use scipy to find p-value.
    t = (mu_diff - mu_0) / (s_diff / np.sqrt(N))
    p = stats.distributions.t.sf(np.abs(t), dof) * 2 # two-sided pvalue
    return t, p


def fit_line(x, y):
    """
    :param x: x coordinates of data points
    :param y: y coordinates of data points
    :return: a, b - slope and intercept of the fitted line;
            idx, idy - indices where x=0, and the fit is not possible
    """

    m = x.shape[0]
    denom = m * np.sum(x ** 2, axis=0) - (np.sum(x, axis=0)) ** 2

    a = (m * np.sum(x * y, axis=0) - np.sum(x, axis=0) * np.sum(y, axis=0)) / denom
    b = (np.sum(x ** 2, axis=0) * np.sum(y, axis=0) - np.sum(x * y, axis=0) * np.sum(x, axis=0)) / denom

    return a, b


def fit_line_no_intercept(x, y):
    a_estimated = np.sum(x * y) / np.sum(x ** 2)

    return a_estimated


def use_welch_test(cnt_layer, history, history_noisy):
    layer_history = np.array([sublist[cnt_layer] for sublist in history])
    layer_history_noisy = np.array([sublist[cnt_layer] for sublist in history_noisy])

    mu1 = np.mean(layer_history, axis=0)
    mu2 = np.mean(layer_history_noisy, axis=0)

    s1 = np.std(layer_history, ddof=0, axis=0)
    s2 = np.std(layer_history_noisy, ddof=0, axis=0)

    N1 = layer_history.shape[0]  # TODO: check this
    N2 = N1

    t_stats, p_vals = welch_t_test(mu1, s1, N1, mu2, s2, N2)

    # print('std min and max: ', np.min(s1), np.max(s1), np.min(s2), np.max(s2))
    # print('mu min and max: ', np.min(mu1), np.max(mu1), np.min(mu2), np.max(mu2))
    # print('p_vals min and max: ', np.min(p_vals), np.max(p_vals))
    return t_stats, p_vals


def use_paired_test(cnt_layer, history, history_noisy):
    layer_history = np.array([sublist[cnt_layer] for sublist in history])
    layer_history_noisy = np.array([sublist[cnt_layer] for sublist in history_noisy])
    mu_diff = np.mean(layer_history - layer_history_noisy, axis=0)
    std_diff = np.std(layer_history - layer_history_noisy, ddof=1, axis=0)
    N1 = layer_history.shape[0]  # config.history_window # TODO: check this
    t_vals, p_vals = paired_t_test(mu_diff, std_diff, N1)

    return t_vals, p_vals
