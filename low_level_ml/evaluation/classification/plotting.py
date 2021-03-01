"""Basic Plotting Utilities for Classification Tasks
"""
import numpy as np
from itertools import cycle
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def compare_classifiers(
        y_true_list,
        y_pred_list,
        label_list,
        weights_list=None,
        fig=None,
        axes=None,
        figsize=(12, 8),
        suptitle=None,
        file_path=None,
        ):
    """Creates Comparison Plots for binary Classifiers

    This function creates four plots that compare different metrics
    for each of the provided classifier predictions.

    Parameters
    ----------
    y_true_list : list of array_like
        A list of the true classification values for each of the provided
        classifier predictions `y_pred_list`-.
        A binary classification is assumed.
        List length: number of classifiers
    y_pred_list : list of array_like
        A list of the classifier scores.
        List length: number of classifiers
    label_list : list of str
        A list of human readable labels for each of the provided classifier
        scores. These are used in the legend. The order must be given in the
        same order as the provided classifier scores `y_pred_list`.
        List length: number of classifiers
    weights_list : None or list of array_like, optional
        If provided, a weighted AUC curve will be computed.
        The shape must match those of `y_true_list` and `y_pred_list`.
        List length: number of classifiers (if provided)
    fig : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If an existing fig is to be used, the corresponding axis `axes` must
        also be set.
    axes : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If existing axes are to be used, the corresponding figure `fig` must
        also be set.
    figsize : tuple, optional
        Only relevant if `fig` and `ax` are None. Defines the figure size.
    suptitle : str, optional
        If provided a figure title will be added.
    file_path : str or None, optional
        If provided, the created plot will be saved to the specified path.

    Returns
    -------
    Figure
        The matplotlib Figure.
    axes
        The matplotlib axes.
        Shape: [2, 2]

    Raises
    ------
    ValueError
        If incorrect inputs for `fig` and `ax` are provided.

    """
    if fig is None or axes is None:
        if fig != axes:
            raise ValueError(
                'You must provide either both `fig` and `axes`'
                ' or neither of them.'
            )
        fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Score distributions
    compare_score_distributions(
        y_true_list=y_true_list,
        y_pred_list=y_pred_list,
        label_list=label_list,
        weights_list=weights_list,
        skip_discrete_classifiers=True,
        title='Score Distribution',
        fig=fig,
        ax=axes[0, 0],
    )

    # ROC curve
    compare_roc_curve(
        y_true_list=y_true_list,
        y_pred_list=y_pred_list,
        label_list=label_list,
        weights_list=weights_list,
        title='ROC Curve with AUC',
        fig=fig,
        ax=axes[0, 1],
    )

    # Precision-Recall curve
    compare_precision_recall_curves(
        y_true_list=y_true_list,
        y_pred_list=y_pred_list,
        label_list=label_list,
        weights_list=weights_list,
        title='Precision-Recall Curve',
        fig=fig,
        ax=axes[1, 0],
    )

    # Precision and recall as a function of classification score
    compare_precision_and_recall(
        y_true_list=y_true_list,
        y_pred_list=y_pred_list,
        label_list=label_list,
        weights_list=weights_list,
        title='Precision and Recall',
        fig=fig,
        ax=axes[1, 1],
    )

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize='xx-large')

    fig.tight_layout()
    if file_path is not None:
        fig.savefig(file_path)

    return fig, axes


def compare_roc_curve(
        y_true_list,
        y_pred_list,
        label_list,
        weights_list=None,
        fig=None,
        ax=None,
        figsize=(6, 4),
        xscale='linear',
        title=None,
        file_path=None,
        legend_kwargs={'fontsize': 'smaller'},
        ):
    """Comparison Plot of the AUC curve

    This function creates a comparison plot of the AUC curve for
    each of the provided classifier predictions.

    Parameters
    ----------
    y_true_list : list of array_like
        A list of the true classification values for each of the provided
        classifier predictions `y_pred_list`-.
        A binary classification is assumed.
        List length: number of classifiers
    y_pred_list : list of array_like
        A list of the classifier scores.
        List length: number of classifiers
    label_list : list of str
        A list of human readable labels for each of the provided classifier
        scores. These are used in the legend. The order must be given in the
        same order as the provided classifier scores `y_pred_list`.
        List length: number of classifiers
    weights_list : None or list of array_like, optional
        If provided, a weighted AUC curve will be computed.
        The shape must match those of `y_true_list` and `y_pred_list`.
        List length: number of classifiers (if provided)
    fig : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If an existing fig is to be used, the corresponding axis `ax` must
        also be set.
    ax : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If an existing ax is to be used, the corresponding figure `fig` must
        also be set.
    figsize : tuple, optional
        Only relevant if `fig` and `ax` are None. Defines the figure size.
    xscale : str, optional
        The scale of the x-axis.
    title : None, optional
        The title of the axis.
    file_path : str or None, optional
        If provided, the created plot will be saved to the specified path.
    legend_kwargs : dict, optional
        Keyword arguments that are passed on to legend.

    Returns
    -------
    Figure
        The matplotlib Figure.
    axis
        The matplotlib axis.

    Raises
    ------
    ValueError
        If incorrect inputs for `fig` and `ax` are provided.
    """
    # create dummy weights if None provided
    if weights_list is None:
        weights_list = [np.ones_like(y) for y in y_true_list]

    # create fig and ax if None provided
    if fig is None or ax is None:
        if fig != ax:
            raise ValueError(
                'You must provide either both `fig` and `ax`'
                ' or neither of them.'
            )
        fig, ax = plt.subplots(figsize=figsize)

    # random guess
    x_random = np.linspace(0., 1, 1000)
    ax.plot(
        x_random, x_random,
        linestyle='--',
        color='0.6',
        label='Random Guess',
    )

    # now loop through classifiers and plot roc
    for y_true, y_pred, label, weights in zip(
            y_true_list, y_pred_list, label_list, weights_list):

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(
            y_true=y_true,
            y_score=y_pred,
            sample_weight=weights,
        )
        roc_auc = auc(fpr, tpr)

        if len(np.unique(y_pred)) <= 2:
            # this is a discrete classifier: True/False
            assert fpr[0] == 0., fpr
            assert tpr[0] == 0., tpr
            assert fpr[2] == 1., fpr
            assert tpr[2] == 1., tpr

            ax.scatter(
                fpr[1], tpr[1],
                marker='+',
                label=label + ' (AUC = {:.2f})'.format(roc_auc),
            )
        else:
            ax.plot(
                fpr, tpr, label=label + ' (AUC = {:.2f})'.format(roc_auc),
            )

    ax.set_xscale(xscale)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(**legend_kwargs)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if file_path is not None:
        fig.savefig(file_path)

    return fig, ax


def compare_precision_recall_curves(
        y_true_list,
        y_pred_list,
        label_list,
        weights_list=None,
        fig=None,
        ax=None,
        figsize=(6, 4),
        xscale='linear',
        yscale='log',
        title=None,
        file_path=None,
        legend_kwargs={'loc': 'upper right', 'fontsize': 'smaller'},
        ):
    """Compare Precision-Recall Curves

    This function creates a comparison plot of the classifier's precision
    and recall curves for each of the provided classifiers.

    Parameters
    ----------
    y_true_list : list of array_like
        A list of the true classification values for each of the provided
        classifier predictions `y_pred_list`-.
        A binary classification is assumed.
        List length: number of classifiers
    y_pred_list : list of array_like
        A list of the classifier scores.
        List length: number of classifiers
    label_list : list of str
        A list of human readable labels for each of the provided classifier
        scores. These are used in the legend. The order must be given in the
        same order as the provided classifier scores `y_pred_list`.
        List length: number of classifiers
    weights_list : None or list of array_like, optional
        If provided, a weighted AUC curve will be computed.
        The shape must match those of `y_true_list` and `y_pred_list`.
        List length: number of classifiers (if provided)
    fig : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If an existing fig is to be used, the corresponding axis `ax` must
        also be set.
    ax : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If an existing ax is to be used, the corresponding figure `fig` must
        also be set.
    figsize : tuple, optional
        Only relevant if `fig` and `ax` are None. Defines the figure size.
    xscale : str, optional
        The scale of the x-axis.
    yscale : str, optional
        The scale of the y-axis.
    title : None, optional
        The title of the axis.
    file_path : str or None, optional
        If provided, the created plot will be saved to the specified path.
    legend_kwargs : dict, optional
        Keyword arguments that are passed on to legend.

    Returns
    -------
    Figure
        The matplotlib Figure.
    axis
        The matplotlib axis.

    Raises
    ------
    ValueError
        If incorrect inputs for `fig` and `ax` are provided.
    """
    # create dummy weights if None provided
    if weights_list is None:
        weights_list = [np.ones_like(y) for y in y_true_list]

    # create fig and ax if None provided
    if fig is None or ax is None:
        if fig != ax:
            raise ValueError(
                'You must provide either both `fig` and `ax`'
                ' or neither of them.'
            )
        fig, ax = plt.subplots(figsize=figsize)

    # now loop through classifiers and plot roc
    for y_true, y_pred, label, weights in zip(
            y_true_list, y_pred_list, label_list, weights_list):

        # Compute precision and recall
        precision, recall, thresholds = precision_recall_curve(
            y_true=y_true,
            probas_pred=y_pred,
            sample_weight=weights,
        )
        average_precision = average_precision_score(
            y_true=y_true,
            y_score=y_pred,
            sample_weight=weights,
        )

        if len(np.unique(y_pred)) <= 2:
            # this is a discrete classifier: True/False

            ax.scatter(
                recall[1], precision[1],
                marker='+',
                label=label + ' (AP = {:.2f})'.format(average_precision),
            )
        else:
            ax.plot(
                recall, precision,
                label=label + ' (AP = {:.2f})'.format(average_precision),
            )

    # draw faint lines at 10% and 1%
    ax.axhline(0.1, ls='--', color='0.8')
    ax.axhline(1e-2, ls='--', color='0.8')

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel(r'Recall: $S / S_{total}$')
    ax.set_ylabel(r'Precision: $S / (S + B)$')

    ax.legend(**legend_kwargs)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if file_path is not None:
        fig.savefig(file_path)

    return fig, ax


def compare_precision_and_recall(
        y_true_list,
        y_pred_list,
        label_list,
        weights_list=None,
        fig=None,
        ax=None,
        figsize=(6, 4),
        xscale='linear',
        yscale='log',
        title=None,
        file_path=None,
        legend_kwargs={'loc': 'lower center', 'fontsize': 'smaller'},
        ):
    """Compare Precision and Recall as a function of Classification Score Cut

    This function creates a comparison plot of the classifier's precision
    and recall for given score cuts for each of the provided classifiers.

    Parameters
    ----------
    y_true_list : list of array_like
        A list of the true classification values for each of the provided
        classifier predictions `y_pred_list`-.
        A binary classification is assumed.
        List length: number of classifiers
    y_pred_list : list of array_like
        A list of the classifier scores.
        List length: number of classifiers
    label_list : list of str
        A list of human readable labels for each of the provided classifier
        scores. These are used in the legend. The order must be given in the
        same order as the provided classifier scores `y_pred_list`.
        List length: number of classifiers
    weights_list : None or list of array_like, optional
        If provided, a weighted AUC curve will be computed.
        The shape must match those of `y_true_list` and `y_pred_list`.
        List length: number of classifiers (if provided)
    fig : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If an existing fig is to be used, the corresponding axis `ax` must
        also be set.
    ax : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If an existing ax is to be used, the corresponding figure `fig` must
        also be set.
    figsize : tuple, optional
        Only relevant if `fig` and `ax` are None. Defines the figure size.
    xscale : str, optional
        The scale of the x-axis.
    yscale : str, optional
        The scale of the y-axis.
    title : None, optional
        The title of the axis.
    file_path : str or None, optional
        If provided, the created plot will be saved to the specified path.
    legend_kwargs : dict, optional
        Keyword arguments that are passed on to legend.

    Returns
    -------
    Figure
        The matplotlib Figure.
    axis
        The matplotlib axis.

    Raises
    ------
    ValueError
        If incorrect inputs for `fig` and `ax` are provided.
    """
    # create dummy weights if None provided
    if weights_list is None:
        weights_list = [np.ones_like(y) for y in y_true_list]

    # create fig and ax if None provided
    if fig is None or ax is None:
        if fig != ax:
            raise ValueError(
                'You must provide either both `fig` and `ax`'
                ' or neither of them.'
            )
        fig, ax = plt.subplots(figsize=figsize)

    # instantiate a second axes that shares the same x-axis
    twin_ax = ax.twinx()

    # dummy lines for labeling
    ax.plot(np.inf, ls='--', color='0.0', label='Recall')
    ax.plot(np.inf, ls='-', color='0.0', label='Precision')

    color_cycler_dis = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    color_cycler = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # now loop through classifiers and plot roc
    for y_true, y_pred, label, weights in zip(
            y_true_list, y_pred_list, label_list, weights_list):

        # Compute precision and recall
        precision, recall, thresholds = precision_recall_curve(
            y_true=y_true,
            probas_pred=y_pred,
            sample_weight=weights,
        )

        if len(np.unique(y_pred)) <= 2:
            # this is a discrete classifier: True/False
            color = next(color_cycler_dis)

            # ax.scatter(
            #     0.5, precision[1],
            #     marker='+',
            #     color=color,
            #     ls='-',
            #     label=label,
            # )
            # twin_ax.scatter(
            #     0.5, recall[1],
            #     marker='+',
            #     color=color,
            #     ls='--',
            # )
        else:
            color = next(color_cycler)

            ax.plot(
                thresholds, precision[:-1],
                label=label,
                color=color,
                ls='-',
            )
            twin_ax.plot(
                thresholds, recall[:-1],
                color=color,
                ls='--',
            )

    # draw faint lines at 10% and 1%
    ax.axhline(0.1, ls='-', color='0.9')
    ax.axhline(1e-2, ls='-', color='0.9')
    twin_ax.axhline(0.1, ls='--', color='0.9')
    twin_ax.axhline(1e-2, ls='--', color='0.9')

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    twin_ax.set_yscale(yscale)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('Classification Score')
    ax.set_ylabel(r'Precision: $S / (S + B)$')
    twin_ax.set_ylabel(r'Recall: $S / S_{total}$')

    ax.legend(**legend_kwargs)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if file_path is not None:
        fig.savefig(file_path)

    return fig, ax, twin_ax


def compare_score_distributions(
        y_true_list,
        y_pred_list,
        label_list,
        weights_list=None,
        skip_discrete_classifiers=False,
        bins=np.linspace(0, 1, 50),
        fig=None,
        ax=None,
        figsize=(6, 4),
        yscale='log',
        title=None,
        file_path=None,
        legend_kwargs={'loc': 'upper center', 'fontsize': 'smaller'},
        ):
    """Compare Score Distributions

    This function creates a comparison plot of the classifier score
    distribution for each of the provided classifiers.

    Parameters
    ----------
    y_true_list : list of array_like
        A list of the true classification values for each of the provided
        classifier predictions `y_pred_list`-.
        A binary classification is assumed.
        List length: number of classifiers
    y_pred_list : list of array_like
        A list of the classifier scores.
        List length: number of classifiers
    label_list : list of str
        A list of human readable labels for each of the provided classifier
        scores. These are used in the legend. The order must be given in the
        same order as the provided classifier scores `y_pred_list`.
        List length: number of classifiers
    weights_list : None or list of array_like, optional
        If provided, a weighted AUC curve will be computed.
        The shape must match those of `y_true_list` and `y_pred_list`.
        List length: number of classifiers (if provided)
    skip_discrete_classifiers : bool, optional
        If True, discrete classifiers that only output two values will
        be skipped and not plotted.
    bins : array_like, optional
        The bins to use for the score distribution histogram.
    fig : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If an existing fig is to be used, the corresponding axis `ax` must
        also be set.
    ax : None, optional
        If None a new Matplotlib Figure and axis will be created.
        If an existing ax is to be used, the corresponding figure `fig` must
        also be set.
    figsize : tuple, optional
        Only relevant if `fig` and `ax` are None. Defines the figure size.
    yscale : str, optional
        The scale of the y-axis.
    title : None, optional
        The title of the axis.
    file_path : str or None, optional
        If provided, the created plot will be saved to the specified path.
    legend_kwargs : dict, optional
        Keyword arguments that are passed on to legend.

    Returns
    -------
    Figure
        The matplotlib Figure.
    axis
        The matplotlib axis.

    Raises
    ------
    ValueError
        If incorrect inputs for `fig` and `ax` are provided.
    """
    # create dummy weights if None provided
    if weights_list is None:
        weights_list = [np.ones_like(y) for y in y_true_list]

    # create fig and ax if None provided
    if fig is None or ax is None:
        if fig != ax:
            raise ValueError(
                'You must provide either both `fig` and `ax`'
                ' or neither of them.'
            )
        fig, ax = plt.subplots(figsize=figsize)

    color_cycler = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # now loop through classifiers and plot distribution
    for y_true, y_pred, label, weights in zip(
            y_true_list, y_pred_list, label_list, weights_list):

        # plot signal
        if not (skip_discrete_classifiers and len(np.unique(y_pred)) <= 2):
            color = next(color_cycler)
            mask_signal = y_true > 0.5
            ax.hist(
                y_pred[mask_signal],
                histtype='step',
                ls='-',
                color=color,
                label=label,
                bins=bins,
            )

            # background
            ax.hist(
                y_pred[~mask_signal],
                histtype='step',
                ls='--',
                color=color,
                bins=bins,
            )

    # dummy lines for signal / background
    ax.plot(np.inf, ls='--', color='0.0', label='Background')
    ax.plot(np.inf, ls='-', color='0.0', label='Signal')

    ax.set_yscale(yscale)
    ax.set_xlabel('Classifier Score')
    ax.set_ylabel('Number of Events')
    ax.legend(**legend_kwargs)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if file_path is not None:
        fig.savefig(file_path)

    return fig, ax
