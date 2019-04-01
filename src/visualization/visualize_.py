import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def evaluate(results, accuracy_baseline, f1_baseline):
    figure, axes = plt.subplot(2, 3, figsize=(11, 7))