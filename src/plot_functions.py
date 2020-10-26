import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.gridspec as gridspec
import itertools
from mlxtend.plotting import plot_decision_regions
import matplotlib.ticker as ticker


def plot_distribution(x_raw, y_raw, x_test, y_test):
    df_raw = pd.DataFrame(dict(x=x_raw[:,0], y=x_raw[:,1], label=y_raw))
    df_test = pd.DataFrame(dict(x=x_test[:,0], y=x_test[:,1], label=y_test))
    colors = {0:'#ef8a62', 1:'#67a9cf'}
    fig, ax = plt.subplots(figsize=(10,4), dpi=100, nrows=1, ncols=2)
    grouped_raw = df_raw.groupby('label')
    grouped_test = df_test.groupby('label')
    for key, group in grouped_raw:
        group.plot(ax=ax[0], kind='scatter', x='x', y='y', label=key, color=colors[key])
    for key, group in grouped_test:
        group.plot(ax=ax[1], kind='scatter', x='x', y='y', label=key, color=colors[key])
    ax[0].set_title('Dados Bruto')
    ax[0].set_xlabel('X1')
    ax[0].set_ylabel('X2')
    ax[1].set_title('Dados de Teste')
    ax[1].set_xlabel('X1')
    ax[1].set_ylabel('X2')
    fig.tight_layout()
    plt.show()


def plot_class_predictions(x, y, learner):

    predictions = learner.predict(x)
    is_correct = (predictions == y)
    unqueried_score = learner.score(x, y)
    x_component, y_component = x[:, 0], x[:, 1]

    # Plot our classification results.
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.scatter(x=x_component[is_correct],  y=y_component[is_correct],  c='g', marker='+', label='Correct')
    ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect')
    ax.legend(loc='lower right')
    ax.set_title("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=unqueried_score))
    plt.show()


def plot_decision_boundary(x, y, active_model, random):

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(16, 14))
    labels = ['arctive learning',
            'random']
    for clf, lab, grd in zip([active_model, random],
                            labels,
                            itertools.product([0, 1],
                            repeat=2)):
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=x, y=y,
                                    clf=clf, legend=2)
        plt.title(lab)

    plt.show()


def plot_performance(history_random, history_active):
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

    ax.plot(history_active)
    line1 = ax.scatter(range(len(history_active)), history_active, s=13)
    ax.plot(history_random)
    line2 = ax.scatter(range(len(history_random)), history_random, s=13)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20, integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)

    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')
    ax.legend((line1, line2), ('Uncertainty sampling', 'Random'))

    plt.show()