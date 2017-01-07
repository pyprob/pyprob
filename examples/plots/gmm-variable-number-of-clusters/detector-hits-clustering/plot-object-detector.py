import matplotlib
import matplotlib.pylab as plt
import numpy as np
import numpy.linalg
import csv
import sys, getopt
from scipy.stats import chi2
from PIL import Image

def tableau20(k):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    return tableau20[k]

def marker_style(k):
    markers = ['x', 'o', '*']
    return markers[k % len(markers)]

def chunks(lst, n):
    n = max(1, n)
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def read_data(data_filename):
    with open(data_filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            field = row[0]

            if field == 'num-samples':
                num_samples = int(row[1])
            elif field == 'dimension':
                dimension = int(row[1])
            elif field == 'data':
                data = np.array(chunks([float(x) for x in row[1:]], dimension))
            elif field == 'clusters':
                clusters = np.array([int(x) for x in row[1:]])
            elif field == 'cluster-probs':
                cluster_probs = np.array([float(x) for x in row[1:]])
            elif field == 'means':
                means = np.array(chunks([float(x) for x in row[1:]], dimension))
            elif field == 'vars':
                variances = np.array([float(x) for x in row[1:]])
            else:
                print('not found')
    return num_samples, dimension, data, clusters, cluster_probs, means, variances

def get_ellipse(mean, cov, num_points, confidence, cluster_prob = 1.0):
    """
    Return [num_points * 2] np.array whose rows correspond to the point on the ellipse which encloses confidence (0 < confidence < 1) of the probability mass of a bivariate normal distribution parametrised by mean and cov.

    See https://bitbucket.org/tuananhle/smc-data-driven/src/830819061350d3fe7e1e77dabb72a23c4bb7782a/src/ddpmo_lein/plotting/getEllipse.m?at=april11&fileviewer=file-view-default.
    """
    eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(cov))
    k = chi2.ppf(confidence, 2)
    theta = np.reshape(np.linspace(0, 2 * np.pi, num = num_points), (-1, 1))
    a = np.sqrt(k / eigenvals[0])
    b = np.sqrt(k / eigenvals[1])
    y = np.concatenate([a * np.sin(theta), b * np.cos(theta)], axis = 1)
    ellipse = np.dot(y, np.linalg.inv(eigenvecs)) + np.tile(mean, [num_points, 1])
    return ellipse

def plot_mvn(ax, mean, cov, cluster_prob = 0, param_dict = {}):
    ax.plot(*zip(*get_ellipse(mean, cov, 100, 0.66)), **param_dict)
    ax.plot(*zip(*get_ellipse(mean, cov, 100, 0.95)), **param_dict)
    out = ax.plot(*zip(*get_ellipse(mean, cov, 100, 0.99, cluster_prob)), **param_dict)
    return out[0]

def make_plots(ax, data, image, imagewidth, imageheight):
    num_samples, dimension, data, clusters, cluster_probs, means, variances = data

    ax.imshow(image,alpha=0.6)
    # set up axes
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_linestyle('dotted')
    # ax.spines["bottom"].set_linewidth(0.5)
    # ax.spines["bottom"].set_color('black')
    # ax.spines["bottom"].set_alpha(0.3)
    # ax.yaxis.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    # ax.xaxis.grid(b=True, which='major', linestyle='dotted', lw=0.5, color='black', alpha=0.3)
    # ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    ax.set_xlim(0, imagewidth)
    # ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_ylim(imageheight, 0)
    # ax.set_yticks([-2, -1, 0, 1, 2])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # plot
    labels = []
    handles = []

    # data_points = ax.scatter(*zip(*data))
    for i in range(len(data)):
        mindist = 100000000
        # clusters[i] = 0
        for j in range(len(means)):
            dist = np.linalg.norm(data[i] - means[j])
            if dist < mindist:
                mindist = dist
                clusters[i] = j

        ax.scatter([((data[i][0] + 1) / 2)  * imagewidth], [((data[i][1] + 1) / 2) * imageheight], marker=marker_style(clusters[i]), color=tableau20(clusters[i]))
    # labels.append('Data')
    # handles.append(matplotlib.lines.Line2D([0], [0], color='black', marker='x', linestyle='None'))

    for i in range(len(means)):
        plot_mvn(ax, np.array([((means[i][0] + 1) / 2) * imagewidth, ((means[i][1] + 1) / 2) * imageheight]), np.array([[variances[i] * imagewidth * imagewidth / 4, 0],[0, variances[i] * imageheight * imageheight /4]]), cluster_probs[i], {'color': tableau20(i), 'linestyle': 'solid', 'linewidth': 2})
        # labels.append('Cluster ' + str(i + 1))
        # handles.append(matplotlib.lines.Line2D([0], [0], color=tableau20(i)))

    # ax.legend(handles, labels, frameon=False, bbox_to_anchor=(0, -0.08, 1, 0), loc='upper center', ncol=3)

    return ax

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "i:m:o:", ["datafile", "imagefile", "plotfile"])
    except getopt.GetoptError:
        print("plot.py -i <datafile> -m <imagefile> -o <output>")
        sys.exit(2)

    for opt, arg in opts:
        print(opt)
        print(arg)
        if opt in ("-i", "--datafile"):
            data_filename_arg = arg
        elif opt in ("-m", "--imagefile"):
            image_filename_arg = arg
        elif opt in ("-o", "--plotfile"):
            plot_filename = arg

    data_filenames = data_filename_arg.split(',')
    image_filenames = image_filename_arg.split(',')

    # set up figure
    matplotlib.rc('font', size=12)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    (imagewidth, imageheight) = Image.open(image_filenames[0]).size

    # plot
    data = read_data(data_filenames[0])
    image = plt.imread(image_filenames[0])
    make_plots(ax, data, image, imagewidth, imageheight)

    # axes postprocessing
    # ax.legend(handles=handles, loc='lower left', bbox_to_anchor=[0, 0], frameon=False, ncol=4, fontsize=8)

    # print figure
    fig.savefig(plot_filename, bbox_inches='tight')
    print("Plot saved to " + plot_filename)

if __name__ == "__main__":
    main(sys.argv[1:])
