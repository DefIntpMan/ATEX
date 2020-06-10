import matplotlib.pyplot as plt
import numpy as np

def plot_simple():
    x = [0.02, 0.04, 0.06, 0.08]
    y_org = [0.70, 0.636, 0.586, 0.545]
    y_def = [0.76, 0.688, 0.635, 0.592]
    plt.ylabel('top 50 intersection')
    plt.xlabel('epsilon')
    plt.plot(x, y_org, "--bo", label="no defense")
    plt.plot(x, y_def, "--ro", label="defense")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    plot_simple()