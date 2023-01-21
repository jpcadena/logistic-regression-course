"""
Visualization script
"""
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from core.config import FIG_SIZE, PALETTE, RE_PATTERN, RE_REPL, FONT_SIZE


def plot_count(dataframe: pd.DataFrame, hue: str) -> None:
    """
    This method plots the counts of observations from the given variables
    :param dataframe: dataframe containing tweets info
    :type dataframe: pd.DataFrame
    :return: None
    :rtype: NoneType
    """
    plot_iterator: int = 1
    variables = dataframe.select_dtypes(include='category')
    for i in variables:
        plt.figure(figsize=FIG_SIZE)
        sns.countplot(x=dataframe[i], hue=dataframe[hue], palette=PALETTE)
        label = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=i)
        plt.xlabel(label, fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plt.title(f'Count-plot for {label}')
        plot_iterator += 1
        plt.tight_layout()
        plt.savefig(f'reports/figures/discrete_{i}.png')
        plt.show()


def plot_distribution(df_column: pd.Series, color: str) -> None:
    """
    This method plots the distribution of the given quantitative
     continuous variable
    :param df_column: Single column
    :type df_column: pd.Series
    :param color: color for the distribution
    :type color: str
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    label: str = re.sub(
        pattern=RE_PATTERN, repl=RE_REPL, string=str(df_column.name))
    dist_plot = sns.displot(x=df_column, kde=True, color=color, height=8,
                            aspect=1.5)
    plt.title('Distribution Plot for ' + label)
    plt.xlabel(label, fontsize=FONT_SIZE)
    plt.ylabel('Frequency', fontsize=FONT_SIZE)
    dist_plot.fig.tight_layout()
    plt.savefig('reports/figures/' + str(df_column.name) + '.png')
    plt.show()


def boxplot_dist(
        dataframe: pd.DataFrame, first_variable: str, second_variable: str
) -> None:
    """
    This method plots the distribution of the first variable data
    in regard to the second variable data in a boxplot
    :param dataframe: data to use for plot
    :type dataframe: pd.DataFrame
    :param first_variable: first variable to plot
    :type first_variable: str
    :param second_variable: second variable to plot
    :type second_variable: str
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    x_label: str = re.sub(
        pattern=RE_PATTERN, repl=RE_REPL, string=first_variable)
    y_label: str = re.sub(
        pattern=RE_PATTERN, repl=RE_REPL, string=second_variable)
    sns.boxplot(x=first_variable, y=second_variable, data=dataframe,
                palette=PALETTE)
    plt.title(x_label + ' in regards to ' + y_label, fontsize=FONT_SIZE)
    plt.xlabel(x_label, fontsize=FONT_SIZE)
    plt.ylabel(y_label, fontsize=FONT_SIZE)
    plt.savefig(
        f'reports/figures/discrete_{first_variable}_{second_variable}.png')
    plt.show()


def plot_scatter(dataframe: pd.DataFrame, x: str, y: str, hue: str) -> None:
    """
    This method plots the relationship between x and y for hue subset
    :param dataframe: dataframe containing tweets
    :type dataframe: pd.DataFrame
    :param x: x-axis column name from dataframe
    :type x: str
    :param y: y-axis column name from dataframe
    :type y: str
    :param hue: grouping variable to filter plot
    :type hue: str
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    sns.scatterplot(x=x, data=dataframe, y=y, hue=hue,
                    palette=PALETTE)
    label: str = re.sub(pattern=RE_PATTERN, repl=RE_REPL, string=y)
    plt.title(f'{x} Wise {label} Distribution')
    print(dataframe[[x, y]].corr())
    plt.savefig(f'reports/figures/{x}_{y}_{hue}.png')
    plt.show()


def plot_heatmap(dataframe: pd.DataFrame) -> None:
    """
    Plot heatmap to analyze correlation between features
    :param dataframe: dataframe containing tweets
    :type dataframe: pd.DataFrame
    :return: None
    :rtype: NoneType
    """
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(data=dataframe.corr(), annot=True, cmap="RdYlGn")
    plt.title('Heatmap showing correlations among columns', fontsize=FONT_SIZE)
    plt.savefig('reports/figures/correlations_heatmap.png')
    plt.show()


def plot_confusion_matrix(
        confusion_matrix: np.ndarray, name: str = 'logistic_regression'
) -> None:
    sns.heatmap(
        confusion_matrix,
        annot=True,
        cmap='YlGnBu',
        cbar=False,
        square=True,
        fmt="d"
    )
    plt.title(f'Confusion Matrix as Heatmap for {name}', fontsize=FONT_SIZE)
    plt.ylabel('Real Label')
    plt.xlabel('Predicted Label')
    plt.savefig('reports/figures/confusion_matrix_' + name + '.png')
    plt.show()


def plot_weights(
        logistic_regression: LogisticRegression, columns_names: list[str],
        name: str = 'logistic_regression') -> None:
    weights: pd.Series = pd.Series(
        logistic_regression.coef_[0],
        index=columns_names).sort_values(ascending=False)
    plt.figure(figsize=FIG_SIZE)
    weights.plot(kind='bar')
    plt.savefig('reports/figures/bar_weights_' + name + '.png')
    plt.show()


def plot_roc_curve(fpr, tpr, auc: float) -> None:
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.title('Receiver Operating Characteristic Curve', fontsize=FONT_SIZE)
    plt.legend(loc=4)
    plt.savefig('reports/figures/roc_curve.png')
    plt.show()
