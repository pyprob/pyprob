import matplotlib
import matplotlib.pylab as plt
import numpy as np
import numpy.linalg
import csv
import sys, getopt
import matplotlib.mlab as mlab

def chunks(lst, n):
    n = max(1, n)
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def read_data(data_filename):
    with open(data_filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            field = row[0]

            if field == 'num-reruns':
                num_reruns = int(row[1])
            elif field == 'num-samples':
                num_samples = int(row[1])
            elif field == 'dimension':
                dimension = int(row[1])
            elif field == 'data':
                data = np.array(chunks([float(x) for x in row[1:]], dimension))
            elif field == 'means-MAP-list':
                list_of_means_MAP = np.array(chunks([float(x) for x in row[1:]], dimension))
            else:
                print('not found')
    return num_reruns, num_samples, dimension, data, list_of_means_MAP

def make_plots(ax, data, kde_sigma, title):
    num_reruns, num_samples, dimension, data, list_of_means_MAP = data

    # set up axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # ax.spines["bottom"].set_linestyle('dotted')
    # ax.spines["bottom"].set_linewidth(0.5)
    # ax.spines["bottom"].set_color('black')
    # ax.spines["bottom"].set_alpha(0.0)
    ax.yaxis.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.xaxis.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", left="off", right="off", labelleft="off")
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    # ax.tick_params(labelleft=False)
    # ax.tick_params(labelbottom=False)
    # ax.set_title(title)

    # plot
    labels = []
    handles = []

    for i in range(len(data)):
        ax.scatter([data[i][0]], [data[i][1]], marker='.', color='black', s=0.5)

    delta = 0.025
    x = np.arange(-1.0, 1.0, delta)
    y = np.arange(-1.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))

    for i in range(len(list_of_means_MAP)):
        mean = list_of_means_MAP[i]
        Z = Z + mlab.bivariate_normal(X, Y, kde_sigma, kde_sigma, mean[0], mean[1])

    cs = ax.contour(X, Y, Z)
    # ax.clabel(cs, inline=1, fontsize=10)

    return ax

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "o:", [ "plotfile"])
    except getopt.GetoptError:
        print("plot.py -o <output>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-o", "--plotfile"):
            plot_filename = arg

    num_particles_seq = [1, 10, 100, 1000, 10000]
    smc_titles = ['smc ' + str(x) for x in num_particles_seq]
    smc_filenames = ['smc-' + str(x) + '.csv' for x in num_particles_seq]
    csis_titles = ['csis ' + str(x) for x in num_particles_seq]
    csis_filenames = ['csis-' + str(x) + '.csv' for x in num_particles_seq]

    # set up figure
    matplotlib.rc('font', size=6)
    fig = plt.figure(figsize=(8, 2.0))
    num_plots = len(smc_titles)
    kde_sigma = 0.08

    # smc plots
    for i in range(num_plots):
        ax = fig.add_subplot(2, num_plots, i + 1)

        # plot
        data = read_data(smc_filenames[i])
        make_plots(ax, data, kde_sigma, smc_titles[i])
        # ax.set_title(num_particles_seq[i])

    # csis plots
    for i in range(num_plots):
        ax = fig.add_subplot(2, num_plots, i + 1 + num_plots)

        # plot
        data = read_data(csis_filenames[i])
        make_plots(ax, data, kde_sigma, csis_titles[i])


    # print figure
    fig.savefig(plot_filename, bbox_inches='tight')
    print("Plot saved to " + plot_filename)

if __name__ == "__main__":
    main(sys.argv[1:])
