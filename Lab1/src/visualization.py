import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



def vis_perplexity(perplexity):
    """
    Visualize the perplexity distribution of the model on the test dataset.
    """
    print("Perplexity mean:", np.mean(perplexity))
    # 中位数
    print("Perplexity median:", np.median(perplexity))
    try:
        plt.hist(perplexity, bins=100)
        plt.xlabel('Perplexity')
        plt.ylabel('Frequency')
        plt.title('Perplexity Distribution')
        plt.show()
    except Exception as e:
        print(e)
        print("No perplexity to visualize")

def vis_all_perplexity(results_perplexity):
    """
    Visualize the perplexity distribution of the model on the test dataset.
    """

    for idx in range(2):
        print("Test dataset", idx + 1)
        for prob_func in ["prob", "add_delta_prob", "good_turing_prob"]:
            title_desc = {
                "prob": "Without Smoothing",
                "add_delta_prob": "Add-1 Smoothing",
                "good_turing_prob": "Good-Turing Smoothing"
            }
            colors = ['', '', 'b', 'orange', 'r']
            for n_order in [2,3]:
                perplexity = results_perplexity[(n_order, idx, prob_func)]
                if prob_func == "prob" and (n_order != 2 or idx != 1):
                    continue
                else:
                    sns.histplot(data=perplexity, bins=20, color=colors[n_order], kde=True, label=f"n={n_order}", stat="density",thresh=100)
            plt.legend(['n=2', 'n=3'])
            plt.xlabel('Perplexity (%s)'%title_desc[prob_func])
            plt.ylabel('Probability Density')
            plt.title('Perplexity Distribution')
            plt.savefig(f"tmp/result/perplexity_{idx+1}_{prob_func}.png")








