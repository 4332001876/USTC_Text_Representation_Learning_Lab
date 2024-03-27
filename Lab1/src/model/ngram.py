from dataloader import NgramDataset
from collections import Counter
import random
import numpy as np
from config import Config
import statsmodels.api as sm

"""class NgramModelFreq:
    def __init__(self, n, ngram_dataset: NgramDataset):
        assert n > 1
        self.vocab = list(self.ngram_dataset.vocab)
        self.n = n
        self.ngram_dataset = ngram_dataset
        self.ngram_dataset.set_n(n)
        self.ngram_freq, self.n_1_gram_freq = self.build_ngram()

    def build_ngram(self):
        ngram_list = []
        for idx in range(len(self.ngram_dataset)):
            data = self.ngram_dataset[idx]
            for ngram in data:
                ngram_list.append(ngram)
        ngram_freq = Counter(ngram_list)

        n_1_gram_list = []
        for ngram in ngram_list:
            n_1_gram_list.append(ngram[:-1])
        n_1_gram_freq = Counter(n_1_gram_list)
        return ngram_freq, n_1_gram_freq

    def generate(self, start, max_len=10):
        n = self.n
        res = start
        for _ in range(max_len):
            next_word = self.get_next_word(start) # random.choice(self.ngram_freq[tuple(start)])
            res += next_word
            start = res[-n+1:]
        return res
    
    def get_next_word(self, start):
        # return random.choice(self.ngram_freq[tuple(start)])
        # 根据概率选择下一个词
        if tuple(start) not in self.ngram_freq:
            return random.choice(self.vocab)
        counter = self.ngram_freq[tuple(start)]
        total = sum(counter.values())
        r = random.randint(1, total)
        for k, v in counter.items():
            r -= v
            if r <= 0:
                return k
        return counter.most_common(1)[0][0] # 如果上面的循环没有找到，返回最频繁的词
    
    def prob(self, ngram):
        if ngram[:-1] not in self.ngram_freq:
            return 0
        counter = self.ngram_freq[ngram[:-1]]
        return counter[ngram[-1]] / sum(counter.values())
    
    def add_delta_prob(self, ngram, delta=1):
        if ngram[:-1] not in self.ngram_freq:
            return 1 / len(self.vocab)
        counter = self.ngram_freq[ngram[:-1]]
        return (counter[ngram[-1]]+delta) / (sum(counter.values())+delta*len(self.vocab))
    
    def good_turing_prob(self, ngram):
        pass
    
    def perplexity(self, test_dataset: NgramDataset, prob_func="prob"):
        n = self.n
        total_log_prob = 0
        total_words = 0
        test_dataset.set_n(n)
        for idx in range(len(test_dataset)):
            data = test_dataset[idx]
            for ngram in data:
                if prob_func == "prob":
                    total_log_prob += -1 * self.prob(ngram)
                elif prob_func == "add_delta_prob":
                    total_log_prob += -1 * self.add_delta_prob(ngram)
                total_words += 1
        return 2 ** (total_log_prob / total_words)"""

class NgramModelCond:
    def __init__(self, n, ngram_dataset: NgramDataset):
        assert n > 1
        self.n = n
        self.ngram_dataset = ngram_dataset
        self.vocab = self.ngram_dataset.vocab
        self.ngram_dataset.set_n(n)
        self.ngram_next_word_freq, self.ngram_freq, self.n_1_gram_freq = self.build_ngram()
        self.n_1_good_turing_times_func = self.gen_good_turing_prob(self.n_1_gram_freq, n-1)
        self.n_good_turing_times_func = self.gen_good_turing_prob(self.ngram_freq, n)

    def build_ngram(self):
        ngram_freq = []
        ngram_next_word_freq = {} # {n-1 gram: {next word: freq}}
        for idx in range(len(self.ngram_dataset)):
            data = self.ngram_dataset[idx]
            for ngram in data:
                ngram_freq.append(ngram)
                if ngram[:-1] not in ngram_next_word_freq:
                    ngram_next_word_freq[ngram[:-1]] = [ngram[-1]]
                else:
                    ngram_next_word_freq[ngram[:-1]].append(ngram[-1])
        ngram_freq = Counter(ngram_freq)

        n_1_gram_freq = {} # {n-1 gram: freq}
        for k, v in ngram_next_word_freq.items():
            ngram_next_word_freq[k] = Counter(v)
            n_1_gram_freq[k] = sum(ngram_next_word_freq[k].values())

        n_1_gram_freq = Counter(n_1_gram_freq)
        return ngram_next_word_freq, ngram_freq, n_1_gram_freq

    def generate(self, start, max_len=100):
        n = self.n
        res = start
        for _ in range(max_len):
            next_word = self.get_next_word(start) 
            if next_word == Config.RIGHT_PAD_SYMBOL:
                break
            res += next_word
            start = res[-n+1:]   
        return res
    
    def get_next_word(self, start):
        # return random.choice(self.ngram_next_word_freq[tuple(start)])
        # 根据概率选择下一个词
        if start not in self.ngram_next_word_freq:
            return random.choice(self.vocab)
        counter = self.ngram_next_word_freq[tuple(start)]
        total = sum(counter.values())
        r = random.randint(1, total)
        for k, v in counter.items():
            r -= v
            if r <= 0:
                return k
        return counter.most_common(1)[0][0] # 如果上面的循环没有找到，返回最频繁的词

    def prob(self, ngram):
        if ngram[:-1] not in self.ngram_next_word_freq:
            return 0
        counter = self.ngram_next_word_freq[ngram[:-1]]
        return counter[ngram[-1]] / self.n_1_gram_freq[ngram[:-1]]
    
    def add_delta_prob(self, ngram, delta=1):
        if ngram[:-1] not in self.ngram_next_word_freq:
            return 1 / len(self.vocab)
        counter = self.ngram_next_word_freq[ngram[:-1]]
        return (counter[ngram[-1]] + delta) / (self.n_1_gram_freq[ngram[:-1]] + delta*len(self.vocab))
    
    def gen_good_turing_prob(self, ngram_freq, n_order):
        times_freq = Counter(ngram_freq.values())
        X = np.log(list(times_freq.keys()))
        Y = np.log(list(times_freq.values()))
        linear_fit = np.polyfit(X, Y, 1)

        model = sm.OLS(Y,X)
        results = model.fit()
        print(n_order, "order Good-Turing Linear Fit:")
        print(results.summary())

        N_c_func = lambda c: np.exp(linear_fit[1] + linear_fit[0] * np.log(c))
        n1 = N_c_func(1)
        n = sum(ngram_freq.values())
        default_prob = n1 / n
        # N_c = np.exp(linear_fit[1] + linear_fit[0] * np.log(c))
        return lambda c: (c + 1) * (np.exp(linear_fit[0] * (np.log(c+1)-np.log(c)))) if c>0 else default_prob
    
    def good_turing_prob(self, ngram):
        n_1_freq = self.n_1_gram_freq[ngram[:-1]]
        n_freq = self.ngram_freq[ngram]
        n_1_freq_star = self.n_1_good_turing_times_func(n_1_freq)
        n_freq_star = self.n_good_turing_times_func(n_freq)
        return n_freq_star / n_1_freq_star
        
    def perplexity(self, test_dataset: NgramDataset, prob_func="prob"):
        perplexity_set = []
        test_dataset.set_n(self.n)
        for idx in range(len(test_dataset)):
            data = test_dataset[idx]
            log_probs = []
            for ngram in data:
                if prob_func == "prob":
                    log_probs.append(np.log(self.prob(ngram)))
                elif prob_func == "add_delta_prob":
                    log_probs.append(np.log(self.add_delta_prob(ngram)))
                elif prob_func == "good_turing_prob":
                    log_probs.append(np.log(self.good_turing_prob(ngram)))
            data_perplexity = np.exp(-1 * np.mean(log_probs))
            perplexity_set.append(data_perplexity)
        return perplexity_set

class NGramModelTester:
    def __init__(self) -> None:
        pass