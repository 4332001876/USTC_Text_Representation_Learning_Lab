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
            for n_order in [2,3]:
                perplexity = results_perplexity[(n_order, idx, prob_func)]
                sns.histplot(data=perplexity, bins=20)
            plt.xlabel('Perplexity')
            plt.ylabel('Frequency')
            plt.title('Perplexity Distribution')
            plt.show()







