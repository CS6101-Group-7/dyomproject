"""
Author:
Muhd Assyarul
"""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def generate_loss_graph(history_list,model_list,metric_name,title_name="Loss"):
    """
    :param train_history_list: iterable containing all the training History.
    :param val_history_list: iterable containing all the validation History. Ensure related objects are in the same order as train_history_list
    :param legend_list: iterable containing the model names. Each item will generate name + ' training' and name + ' validation' for legend.
    :metric_name: name of metric used
    :param title_name: title of the graph
    :return: none
    """
    plt.title(title_name)
    for history in history_list:
        plt.plot(history.history[metric_name])
        plt.plot(history.history['val_'+metric_name])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    legend_list=[]

    for model in model_list:
        legend_list.append(model+" train")
        legend_list.append(model+" validation")
    plt.legend(legend_list,loc='upper left')
    plt.show()

def opposite_color(rgbt_list):
    """
    :param rgbt_list: list of color values i.e [red,green,blue,transparency]
    :return: complementary color in the same format
    """
    opposite = [1-x for x in rgbt_list]
    opposite[-1] = rgbt_list[-1]
    return opposite

def generate_confusion_matrix(true_ans_list,pred_ans_list,color_theme='bone',title_name='Confusion Matrix'):
    """
    :param true_ans_list: list of ground truth answers
    :param pred_ans_list: list of predicted ans from model
    :param color_theme: pick a color theme from matplotlib.colors
    """
    fig, ax = plt.subplots()
    cm=confusion_matrix(true_ans_list,[round(x) for x in pred_ans_list])
    im=ax.imshow(cm,cmap=color_theme)
    colours = im.cmap(im.norm(np.unique(cm)))
    colours = np.flipud(colours)
    colours_packed = [[colours[i],colours[i+1]] for i in range(0,len(colours),2)]
    ax.set_xticks(np.arange(0,2))
    ax.set_yticks(np.arange(0,2))
    ax.set_xlabel("predicted value")
    ax.set_ylabel("true value")
    for i in range(0,2):
        for j in range(0,2):
                text = ax.text(j, i, cm[i, j],
                            ha="center", va="center", color=opposite_color(colours_packed[i][j]))
    ax.set_title(title_name)
    fig.tight_layout
    plt.show()


