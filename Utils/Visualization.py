import numpy as np
import matplotlib.pyplot as plt


def plot_orig_recon_imgs(labeled_img, noised_img, iteration, y_true, file_addr):
    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(labeled_img, cmap='gist_gray')
    axarr[0].set_title("Original Image")
    axarr[0].yaxis.set_visible(False)
    axarr[0].xaxis.set_visible(False)
    axarr[1].imshow(noised_img, cmap='gist_gray')
    axarr[1].set_title("Reconstructed Image")
    axarr[1].yaxis.set_visible(False)
    axarr[1].xaxis.set_visible(False)
    fig.suptitle(f"Iteration {iteration}, Image Label: {y_true}")
    fig.savefig(file_addr)
    plt.close()
    # plt.show()


def plot_single(data, title, xlabel="Iteration", ylabel="F1-score"):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_multi(hist_clf_train_f1, hist_clf_test_f1, title, xlabel="Iteration", ylabel="F1-score"):
    plt.plot(hist_clf_train_f1, label="Train")
    plt.plot(hist_clf_test_f1, label="Test")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
