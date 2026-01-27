'''
This file contains core functions used by other modules in the popstatgen package.
'''

# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple

#######################
#### Miscellaneous ####
#######################

def corr(x,y):
    '''
    Computes Pearson correlation coefficient between two vectors x and y.
    Parameters:
        x (1D array): First vector.
        y (1D array): Second vector.
    Returns:
        r (float): Pearson correlation coefficient between x and y.
    '''
    x_norm = (x - x.mean()) / np.sqrt(np.var(x))
    y_norm = (y - y.mean()) / np.sqrt(np.var(y))
    r = (x_norm * y_norm).mean()
    return r

def get_pop_kwargs(i, **kwargs) -> dict:
    '''
    Extracts population-specific parameters from SuperPopulation functions using kwargs.
    Parameters:
        i (int): Index of the population among active populations.
        **kwargs: Additional keyword arguments.
    Returns:
        pop_kwargs (dict): Dictionary of population parameters.
    '''
    pop_kwargs = {}
    # loops through each key-value pair in kwargs
    for key, value in kwargs.items():
        # if the value is a list, try to get the i-th element
        if isinstance(value, list):
            if i < len(value):
                pop_kwargs[key] = value[i]
            else:
                raise IndexError(f"Not enough elements in list for parameter '{key}' to match all populations.")
        # if not a list, use the value directly for all populations
        else:
            pop_kwargs[key] = value
    return pop_kwargs

def report_CI(point_and_se: Tuple[float, float], CI: float = 0.95) -> str:
    '''
    Returns a string with the point estimate and confidence interval.
    Parameters:
        point_and_se (list): Tuple containing the point estimate and standard error (point, se).
        CI (float): Confidence level. Default is 0.95.
    Returns:
        report (str): String with the point estimate and confidence interval.
    '''
    (point, se) = point_and_se
    z = norm.ppf((1 + CI) / 2)  # z-score for the given confidence level
    lower = point - z * se
    upper = point + z * se
    return f'{point:.3f} [{lower:.3f}, {upper:.3f}]'

@staticmethod
def to_bits(n: int, bits: int):
    '''
    Converts integer n to list of bits of length `bits`.
    Parameters:
        n (int): Integer to convert to bits.
        bits (int): Number of bits to convert to.
    Returns:
        bit_list (list): List of bits representing integer n.
    '''
    return [int(b) for b in format(n, f'0{bits}b')]

#######################
#### Visualization ####
#######################
def _get_default_colors(n_lines):
        '''
        Returns a list of colors for plotting lines, cycling through matplotlib's default color cycle.
        Parameters:
            n_lines (int): Number of lines to generate colors for.
        Returns:
            colors (list): List of colors for plotting lines.
        '''
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [c['color'] for c in prop_cycle]
        # Repeat colors if n_lines > length of color cycle
        return [colors[i % len(colors)] for i in range(n_lines)]

def plot_over_time(metrics: np.ndarray,
                   ts: np.ndarray = None,
                   aes: dict = {'title': None, 'xlabel': None, 'ylabel': None},
                   aes_line: dict = {'color': None, 'ls': None, 'labels': None},
                   vlines: np.ndarray = None,
                   legend: bool = True):
    '''
    General function for plotting some metric over time.

    Parameters:
        metrics (1D or 2D array): Metric(s) to plot over time. If a T*K 2D matrix, each column is treated as a different line to plot.
        ts (1D array): Array of time points (generations) corresponding to the metric. If not specified, defaults to the range of generations in the metric.
        aes (dict): Dictionary of aesthetic parameters for the plot.
        aes_line (dict): Dictionary of aesthetic parameters for the lines in the plot.
        vlines (1D array): Array of time points at which to draw vertical lines. Default is None, meaning no vertical lines are drawn.
        legend (bool): Whether to include a legend in the plot for each line. Default is False.
    '''
    if metrics.ndim == 1:
        metrics = metrics.reshape(-1, 1)
    K = metrics.shape[1]
    # fills out aes_line settings
    # color
    if aes_line['color'] is None:
        aes_line['color'] = _get_default_colors(K)
    if type(aes_line['color']) == str:
        aes_line['color'] = [aes_line['color']] * K
    colors = aes_line['color']
    # line style
    if aes_line['ls'] is None:
        aes_line['ls'] = '-'
    if type(aes_line['ls']) == str:
        aes_line['ls'] = [aes_line['ls']] * K
    ls = aes_line['ls']
    # labels
    if aes_line['labels'] is None:
        aes_line['labels'] = [f'Line {j}' for j in range(K)] # not shown
        legend=False
    labels = aes_line['labels']

    if ts is None:
        ts = np.arange(metrics.shape[0])

    # plotting
    plt.figure(figsize=(8, 5))
    # plots lines
    for j in range(K):
        plt.plot(ts, metrics[:, j],
                    color=colors[j],
                    ls=ls[j],
                    label=labels[j])
    # vertical lines
    if vlines is not None:
        for t in vlines:
            plt.axvline(t, ls='--', color='black')
    # labels
    if aes['xlabel'] is not None:
        plt.xlabel(aes['xlabel'])
    if aes['ylabel'] is not None:
        plt.ylabel(aes['ylabel'])
    if aes['title'] is not None:
        plt.title(aes['title'])
    plt.xlim(ts.min(), ts.max())
    plt.ylim(0, 1)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()

