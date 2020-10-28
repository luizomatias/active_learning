import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.gridspec as gridspec
import itertools
from mlxtend.plotting import plot_decision_regions
import matplotlib.ticker as ticker
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


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
    labels = ['active learning',
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



tf_idf = Pipeline(steps=[
    ('tfidf', TfidfVectorizer())
])

def plot_all_distributions(X_raw, y_raw, X_random, X_uncertainty):

    X_sparse = tf_idf.fit_transform(X_raw)
    # Define our LSA transformer and fit it onto our dataset.
    svd = TruncatedSVD(n_components=2, random_state=42)
    data = svd.fit_transform(X_sparse) 
    # Isolate the data we'll need for plotting.
    x_component, y_component = data[:, 0], data[:, 1]

    # Plot our dimensionality-reduced (via LSA) dataset.
    fig, ax = plt.subplots(figsize=(15,7))
    scatter = ax.scatter(x=x_component, y=y_component, c= y_raw, cmap='viridis')
    classes =  ['Baseball', 'Hockey', 'Motorcycles']

    legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=classes,
                    loc="upper left", title="Classes")
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    plt.title('Distribuição Bidimensional Dados Bruto')


    X_sparse_active = tf_idf.fit_transform(X_uncertainty)
    # Define our LSA transformer and fit it onto our dataset.
    svd_active = TruncatedSVD(n_components=2, random_state=42)
    data_active = svd_active.fit_transform(X_sparse_active) 
    # Isolate the data we'll need for plotting.
    x_component_active, y_component_active = data_active[:, 0], data_active[:, 1]

    X_sparse_random = tf_idf.fit_transform(X_random)
    # Define our LSA transformer and fit it onto our dataset.
    svd_random = TruncatedSVD(n_components=2, random_state=42)
    data_random = svd_random.fit_transform(X_sparse_random) 
    # Isolate the data we'll need for plotting.
    x_component_random, y_component_random = data_random[:, 0], data_random[:, 1]
    # Plot our dimensionality-reduced (via LSA) dataset.


    # Plot our dimensionality-reduced (via LSA) dataset.
    fig, ax = plt.subplots(figsize=(20,5), nrows=1, ncols=2)
    scatter = ax[0].scatter(x=x_component, y=y_component, c= '#c2c2c2')
    scatter = ax[0].scatter(x=x_component_random, y=y_component_random, c='blue', cmap='viridis')
    scatter = ax[1].scatter(x=x_component, y=y_component, c= '#c2c2c2')
    scatter = ax[1].scatter(x=x_component_active, y=y_component_active, c='red', cmap='viridis')
    ax[0].set_title('Distribuição Bidimensional Amostra Aleatória', size=15)
    ax[1].set_title('Distribuição Bidimensional Amostra Incerta', size=15)

    plt.show()


def plot_distribution(X, y, title: str):

    X_tfidf = tf_idf.fit_transform(X)
    # Define our LSA transformer and fit it onto our dataset.
    svd = TruncatedSVD(n_components=2, random_state=42)
    data = svd.fit_transform(X_tfidf) 
    # Isolate the data we'll need for plotting.
    x_component, y_component = data[:, 0], data[:, 1]

    # Plot our dimensionality-reduced (via LSA) dataset.
    fig, ax = plt.subplots(figsize=(15,7))
    scatter = ax.scatter(x=x_component, y=y_component, c=y, cmap='viridis')
    classes =  ['Baseball', 'Hockey', 'Motorcycles']

    legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=classes,
                        loc="upper left", title="Classes")
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)

    plt.title(title)
    plt.show()