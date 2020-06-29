import matplotlib
import matplotlib.pyplot as pyplot
import numpy
import datetime


def plot_observations(X, number_of_plots):
    y = range(100)
    try:
        int(number_of_plots)
    except ValueError:
        return False

    for i in range(number_of_plots):
        fig, ax = pyplot.subplots()
        ax.set_title('Data for observation #{0}'.format(i+1))
        ax.plot(y, X[i][0:100])
        # plt.show()
        file = 'plots/plot_{0}.png'.format(datetime.datetime.now())
        fig.savefig(file)
    return True


def plot_precision_recall_curves(recall, precision):
    fig, ax = pyplot.subplots()
    ax.set_title('Precision-Recall Curve')
    ax.plot(recall, precision)
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # plt.show()
    file = 'plots/plot_recision_recall_curve_{0}.png'.format(datetime.datetime.now())
    fig.savefig(file)

    return True