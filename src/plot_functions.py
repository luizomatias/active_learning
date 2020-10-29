import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.gridspec as gridspec
import itertools
from plotnine import *
from mlxtend.plotting import plot_decision_regions
import matplotlib.ticker as ticker
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def plot_distribution(x_raw, y_raw, x_test, y_test):
    
    """
    Plot the distribuition side by side of two dimensional data
    Arguments
    ---------
    x_raw: numpy.ndarray
        the raw data to be plot

    y_raw: numpy.ndarray
        the raw data label plot

    x_test: numpy.ndarray
        the test data to be plot

    y_test: numpy.ndarray
        the test data to be plot

    """
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

    """
    Plot the prediction of the learner active learning strategy
    model in the data

    Arguments
    ---------
    x: numpy.ndarray
        the data to be predicted

    y: numpy.ndarray
        the label of the data to be predicted
    
    learner: modAL.models.ActiveLearner
        the base active learning model 

    """

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


def plot_decision_boundary(x, y, active_model, random_model):

    """
    Plot the decision boudary of two models sibe by side
    Arguments
    ---------
    x: numpy.ndarray
        the raw data to be plot

    y: numpy.ndarray
        the raw data label plot

    active_model: modAL.models.ActiveLearner
        the base active learning model 

    random_model: sklearn.svm.SVC
        the random model

    """

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(16, 14))
    labels = ['Active Learning',
            'Random']
    for clf, lab, grd in zip([active_model, random_model],
                            labels,
                            itertools.product([0, 1],
                            repeat=2)):
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=x, y=y,
                                    clf=clf, legend=2)
        plt.title(lab)

    plt.show()


def plot_performance(history_random: list, history_active: list):

    """
    Plot the performance query iteration x classification accuracy
    of two models

    Arguments
    ---------
    history_random: list
        list of accuracy random model

    history_active:list
        list of accuracy active learning model

    """
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


def dimensionality_reduction_plot(X):

    """
    convert to tf-idf matrix and then Dimensionality reduction
    using truncated SVD (aka LSA)
    ---------
    X: list or pandas.Series
        the data to covert to tf-idf and Dimensionality reduction

    """

    tf_idf = Pipeline(steps=[
    ('tfidf', TfidfVectorizer())
    ])

    X_tfidf = tf_idf.fit_transform(X)
    # Define our LSA transformer and fit it onto our dataset.
    svd = TruncatedSVD(n_components=2, random_state=42)
    data = svd.fit_transform(X_tfidf) 
    # Isolate the data we'll need for plotting.
    x_component, y_component = data[:, 0], data[:, 1]

    return x_component, y_component


def plot_dimensionality_reduction_distribution(X, y, title: str):

    """
    Plot the dimensionality reduction distribution
    Arguments
    ---------
    X: list or pandas.Series
        the data to be plot

    y: list or pandas.Series
        the data label to be plot

    title: str
        title of the plot 

    """

    x_component, y_component = dimensionality_reduction_plot(X)

    d = {'index': x_component, 'col': y_component, 'class': y}

    q_data = pd.DataFrame(data=d)
    q_data['class'] = q_data['class'].astype('category')
    q_data.replace({0: 'Motorcycles', 1:'Baseball', 2:'Hockey'}, inplace=True)

    plot = (ggplot()
     + ggtitle(title)
     + geom_point(q_data, aes(x='index', y='col', color='class'))
     + coord_fixed(ratio=0.5, ylim=(-0.3,0.3))
     + theme_bw()
     + theme(figure_size=(11, 5))
     + labs(x = "X1", y = "X2", aspect_ratio=0.3)
        )
    plot.draw();


def plot_density_distribution(X_raw, X_active, X_random):

    """
    Plot the density distribution of two data side by side
    Arguments
    ---------
    X_raw: list or pandas.Series
        the raw data to be plot

    X_active: list or pandas.Series
        the active learning data to be plot

    X_random: list or pandas.Series
        the random data to be plot

    """

    x_component, y_component = dimensionality_reduction_plot(X_raw)
    x_component_active, y_component_active = dimensionality_reduction_plot(X_active)
    x_component_random, y_component_random = dimensionality_reduction_plot(X_random)

    d = {'index': x_component, 'col': y_component}

    q_data = pd.DataFrame(data=d)

    d_active = {'index': x_component_active, 'col': y_component_active, 'type': 'Active Learning'}
    q_active = pd.DataFrame(data=d_active)
    
    d_random = {'index': x_component_random, 'col': y_component_random, 'type': 'Random'}
    q_random = pd.DataFrame(data=d_random)

    df_concat = pd.concat([q_active, q_random], axis=0)

    plot = (ggplot(df_concat, aes(x='index', y='col', color='type'))
     + ggtitle('Titulo')
     + geom_point(q_data, aes(x='index', y='col'), color='gray', alpha=0.2)
     + geom_point()
     + stat_density_2d()
     + coord_fixed(ratio=0.5, ylim=(-0.3,0.3))
     + facet_wrap('type')
     + theme_bw()
     + theme(figure_size=(11, 14), aspect_ratio=0.9)
     + labs(x = "X1", y = "X2", aspect_ratio=0.3, title= 'Gráfico de Densidade Active Learning x Random')
     + scale_colour_manual({'Active Learning':'red', 'Random':'blue'})
     )
    plot.draw();