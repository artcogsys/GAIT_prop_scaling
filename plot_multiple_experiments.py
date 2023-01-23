from glob import glob
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})
import numpy as np

print(plt.style.available)
plt.style.use('seaborn-colorblind')

def get_exp_folders(exp_folder):
    return glob(exp_folder+'/*')

def plot_line(exp_folders, metric, title, x_label, y_label, logscale=False, show_legend=False, letter=None, ylim=0):

    legend = []
    plt.figure(figsize=(5, 2), dpi=800)
    ax = plt.gca()
    for exp_name in exp_folders:

        if metric == 'cors_per_layer' and exp_name.find('BP') > -1:
            ax._get_lines.get_next_color()
            continue

        with open(os.path.join(exp_name, f'{metric}.txt')) as f:
            if metric == 'cors_per_layer': # Aggregate data per batch for this metric
                cors = []
                for line in f.readlines():
                    cors.append([float(x) for x in line.split(' ')])
                values = np.mean(np.array(cors), axis=1)
            else:
                values = f.readlines()
                values = [float(x)*100 for x in values] # *100 because it is a percentage
            name_tag = os.path.basename(os.path.normpath(exp_name))
            legend.append((name_tag[1:], values))
        plt.plot(values)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if logscale:
        plt.yscale('log')
    if letter is not None:
        plt.figtext(0.0, .9, letter, wrap=True, horizontalalignment='left', fontsize=12)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(ymin=ylim)
    if show_legend:
        ax.legend([x[0] for x in legend], fontsize=8, loc='lower right', bbox_to_anchor=(1, 0.2))
    plt.savefig(os.path.join('plots', f'{title}.png'), bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_bar_with_CI_interval(exp_folders, metric, title, x_label, y_label, letter, ylim=0):

    legend = []
    plt.figure(figsize=(5, 2), dpi=800)
    ax = plt.gca()
    n_exps = len(exp_folders)
    bar_width = .5 / n_exps # the more experiments, the skinnier the bars

    for index, exp_name in enumerate(exp_folders):
        if exp_name.find('BP') > -1:
            ax._get_patches_for_fill.get_next_color()
            continue

        with open(os.path.join(exp_name, f'{metric}.txt')) as f:
            data = []
            values = f.readlines()
            for v in values:
                data.append([float(x) for x in v.split(' ')])
            data = np.array(data)
            mean_values = np.mean(data, axis=0)
            stds = np.std(data, axis=0) #/ np.sqrt(data.shape[0])
            #errs = sems/1.96
            legend.append((os.path.basename(os.path.normpath(exp_name)), mean_values))

        num_values = len(mean_values)
        plt.bar(np.array(range(num_values)) - (index * bar_width) + (n_exps/2 - .5) * bar_width, mean_values, width=bar_width, yerr=stds)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.figtext(0.0, .9, letter, wrap=False, horizontalalignment='left', fontsize=12)
    xticks = list(range(len(mean_values)))
    xlabels = [str(x) for x in xticks]
    xlabels[-1] = 'output'
    plt.xticks(ticks=xticks, labels=xlabels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(ymin=ylim)
    plt.savefig(os.path.join('plots', f'{title}.png'), bbox_inches='tight')
    plt.clf()
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument("-dir", help="Experiment folder.", default='')
args = parser.parse_args()

exp_folder = args.dir

exp_folders = get_exp_folders(exp_folder)
for index, x in enumerate(sorted(exp_folders)):
    print(index, x)

os.makedirs('plots', exist_ok=True)
plot_line(exp_folders, 'train_acc', 'Train_accuracy', 'Epoch', 'train accuracy', letter='C')
plot_line(exp_folders, 'train_acc_top5', 'Train_accuracy_top_5', 'Epoch', 'train accuracy top5', letter='C')
plot_line(exp_folders, 'test_acc', 'Test_accuracy', 'Epoch', 'test accuracy', show_legend=True, letter='D')
plot_line(exp_folders, 'test_acc_top5', 'Test_accuracy_top_5', 'Epoch', 'test accuracy top5', letter='D', show_legend=True)
plot_line(exp_folders, 'cors_per_layer', 'Angles_per_batch', 'train step x100', 'Angle (degrees)', letter='B', ylim=-5)
plot_line(exp_folders, 'test_loss', 'Train_loss', 'Epoch', 'test loss log10', logscale=True)
plot_line(exp_folders, 'train_loss', 'Test_loss', 'Epoch', 'train loss log10', logscale=True)
plot_bar_with_CI_interval(exp_folders, 'cors_per_layer', 'Angles_per_layer', 'Layer number', 'Angle (degrees)', letter='A', ylim=-5)

