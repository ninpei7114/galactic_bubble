import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def make_figure(name, loss_l_list_train, loss_c_list_train, loss_l_list_val, loss_c_list_val,
                train_f1_score, val_f1_score):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_l_list_train, label='loss_l_train')
    ax.plot(loss_c_list_train, label='loss_c_train')
    ax.plot(loss_l_list_val, label='loss_l_val')
    ax.plot(loss_c_list_val, label='loss_c_val')
    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('loss value', fontsize=18)
    ax.grid(linestyle='--')
    plt.minorticks_on()
    plt.legend()
    ax.set_title('confidence & location loss', size=20)
    fig.savefig(name+'/loss_cl.png')


    ## lossの推移
    df = pd.read_csv(name+'/log_output.csv')
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_xticklabels(np.arange(-1, len(df)+2, 2))
    # ax.set_xticklabels([0, 1, 3, 5, 7, 9, 11, 13, 15, 17])
    ax.plot(loss_l_list_train, label='location loss      (train)', linestyle='dotted', color='g')
    ax.plot(loss_c_list_train, label='confidence loss (train)', linestyle='dashdot', color='g')
    ax.plot(loss_l_list_val, label='location loss      (validation)', linestyle='dotted', color='r')
    ax.plot(loss_c_list_val, label='confidence loss (validation)', linestyle='dashdot', color='r')

    ax.plot(df['train_loss'], label='train loss', linewidth=3, color='g')
    ax.plot(df['val_loss'], label='validation loss', linewidth=3, color='r')

    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('loss value', fontsize=18)
    ax.grid(linestyle='--')
    plt.minorticks_on()
    plt.legend()
    ax.set_title('validation loss', size=20)
    fig.savefig(name+'/loss.png')


    ## f1 scoreの推移
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_f1_score, label='train_f1_score')
    ax.plot(val_f1_score, label='val_f1_score')
    ax.set_xlabel('epoch', fontsize=18)
    ax.set_ylabel('score value', fontsize=18)
    ax.grid(linestyle='--')
    plt.minorticks_on()
    plt.legend()
    ax.set_title('train & val F1 score', size=20)
    fig.savefig(name+'/f1_score.png')