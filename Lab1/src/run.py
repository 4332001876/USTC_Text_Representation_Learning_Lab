from dataloader import NgramDataset
from model.ngram import NgramModelCond
from config import Config
from visualization import *

results_perplexity = {}
for n_order in [2,3]:
    train_dataset = NgramDataset(Config.TRAIN_PATH, n_order)
    ngram_model = NgramModelCond(n_order, train_dataset)
    test_dataset_1 = NgramDataset(Config.TEST_1_PATH, n_order)
    test_dataset_2 = NgramDataset(Config.TEST_2_PATH, n_order)
    for idx,test_dataset in enumerate([test_dataset_1, test_dataset_2]):
        print("Test dataset", idx+1)
        for prob_func in ["prob", "add_delta_prob", "good_turing_prob"]:
            print(prob_func)
            perplexity = ngram_model.perplexity(test_dataset, prob_func)
            # vis_perplexity(perplexity)
            results_perplexity[(n_order, idx, prob_func)] = perplexity

vis_all_perplexity(results_perplexity)




    

