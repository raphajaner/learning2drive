import argparse
import glob
import os
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt
import numpy as np


def exp_wma(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha

    scale = 1 / alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale ** r
    offset = data[0] * alpha_rev ** (r + 1)
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def plot_training_curve(csv_files, x_label, y_label):
    """Plot a training curve given a list of csv files with the training data in it but the csv files are recorded in different runs and time steps
    Args:
        csv_files (list): List of csv files with the training data in it
        x_label (str): Label for the x axis
        y_label (str): Label for the y axis
        title (str): Title of the plot
    """
    fig, ax = plt.subplots()

    # colors = ["#A7226E", "#DCEDC2", "#EC2049", "#EC2049", "#F26B38", "#2F9599"]
    # colors = ["magenta", "violet", "teal", "cyan", "brown", "olive"]
    colors = ["violet", "violet", "royalblue", "royalblue", "darkorange", "darkorange"]
    variant_names = ["Gray-BEV FS", "Gray-BEV LSTM", "Multi-BEV FS", "Multi-BEV LSTM", "RGB-BEV FS", "RGB-BEV LSTM"]
    colors = colors[0:len(variant_names)]

    variants = dict()
    for variant_name in variant_names:
        variants[variant_name] = list()

    max_step = 0
    for csv_file in csv_files:
        variant_name = csv_file.split("/")[-2]
        experiment_name = csv_file.split("/")[-1].split(".")[0]
        data = np.genfromtxt(csv_file, dtype=str, delimiter=',', skip_header=1, usecols=(0, 4))

        # from each line remove the first and last character
        data = np.char.strip(data, chars='"')
        data = data.astype(float)
        if np.max(data[:, 0]) > max_step:
            max_step = np.max(data[:, 0])

        # Apply exponential moving average to smooth the curve
        eval_horizon = np.linspace(0, 10 ** 6, 10 ** 4)
        data_horizon = np.interp(eval_horizon, data[:, 0], data[:, 1])

        variants[variant_name].append(data_horizon)

    eval_horizon_plot = eval_horizon[::10]
    for variant_name, color in zip(variants.keys(), colors):
        variant_data = np.empty((10 ** 4, len(variants[variant_name])))
        variant_data_smooth = np.empty((10 ** 4, len(variants[variant_name])))
        for i in range(len(variants[variant_name])):
            variant_data[:, i] = variants[variant_name][i]
            variant_data_smooth[:, i] = exp_wma(variants[variant_name][i], 2000)

        variant_data_mean = np.mean(variant_data_smooth, axis=1)

        # variant_data_avg = copy.deepcopy(data_horizon)
        # variant_data_avg = np.array([eval_horizon, variant_data_avg]).T
        # variant_data_avg[:, 1] = exp_wma(variant_data_avg[:, 1], 150)

        # calculate the mean and standard deviation of the data and plot this
        y_std = np.std(variant_data, axis=1)
        y_std_smooth = exp_wma(y_std, 4000)

        y_std_smooth = y_std_smooth[::10]
        variant_data_mean = variant_data_mean[::10]

        if "FS" in variant_name:
            ax.plot(eval_horizon_plot, variant_data_mean, color=color, label=variant_name, lw=1.5, linestyle='--')
        else:
            ax.plot(eval_horizon_plot, variant_data_mean, color=color, label=variant_name, lw=1.5)
        ax.fill_between(eval_horizon_plot, variant_data_mean - y_std_smooth, variant_data_mean + y_std_smooth,
                        color=color, alpha=0.1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.legend()
    plt.margins(0, tight=True)
    plt.tight_layout()
    plt.savefig(os.path.join("/".join(csv_file.split("/")[0:-2]), 'avg_return.pgf'))
    plt.savefig(os.path.join("/".join(csv_file.split("/")[0:-2]), 'avg_return.pdf'), backend='pgf')
    # plt.show()


# set up argparse
parser = argparse.ArgumentParser(description='Plot training curve')
parser.add_argument('--csv_dir', nargs='+', help='Path to directory with csv files',
                    default='./figures/experiments/')

# parse
args = parser.parse_args()

csv_files = glob.glob(os.path.join(args.csv_dir, '*/*.csv'))

plot_training_curve(csv_files, 'Step', 'Average Return')
