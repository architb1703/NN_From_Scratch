import numpy as np
import matplotlib.pyplot as plt

def plot_metric(hist_arr, curve_label="", ylabel="", title="", sec_hist_arr=None, sec_curve_label=""):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel("Train Step", color=color)
    ax1.set_ylabel(ylabel)
    line1, = ax1.plot(np.arange(len(hist_arr)), hist_arr, label=curve_label, color=color)
    lines = [line1]
    
    if(sec_hist_arr != None):
        ax2 = ax1.twiny()

        color = 'tab:blue'
        ax2.set_xlabel('Val Step', color=color)
        line2, = ax2.plot(np.arange(len(sec_hist_arr)), sec_hist_arr, label=sec_curve_label, color=color)
        lines.append(line2)
    
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    fig.tight_layout()
    plt.show()