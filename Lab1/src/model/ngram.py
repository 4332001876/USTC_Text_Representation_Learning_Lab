from dataloader import NgramDataset
from collections import Counter
import random
from config import Config

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
        self.vocab = self.ngram_dataset.vocab
        self.n = n
        self.ngram_dataset = ngram_dataset
        self.ngram_dataset.set_n(n)
        self.ngram_freq, self.n_1_gram_freq = self.build_ngram()

    def build_ngram(self):
        ngram_freq = {} # {n-1 gram: {next word: freq}}
        for idx in range(len(self.ngram_dataset)):
            data = self.ngram_dataset[idx]
            for ngram in data:
                if ngram[:-1] not in ngram_freq:
                    ngram_freq[ngram[:-1]] = [ngram[-1]]
                else:
                    ngram_freq[ngram[:-1]].append(ngram[-1])

        n_1_gram_freq = {} # {n-1 gram: freq}
        for k, v in ngram_freq.items():
            ngram_freq[k] = Counter(v)
            n_1_gram_freq[k] = sum(ngram_freq[k].values())

        n_1_gram_freq = Counter(n_1_gram_freq)
        return ngram_freq, n_1_gram_freq

    def generate(self, start, max_len=100):
        n = self.n
        res = start
        for _ in range(max_len):
            next_word = self.get_next_word(start) # random.choice(self.ngram_freq[tuple(start)])
            if next_word == Config.RIGHT_PAD_SYMBOL:
                break
            res += next_word
            start = res[-n+1:]   
        return res
    
    def get_next_word(self, start):
        # return random.choice(self.ngram_freq[tuple(start)])
        # 根据概率选择下一个词
        if start not in self.ngram_freq:
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
        return counter[ngram[-1]] / self.n_1_gram_freq[ngram[:-1]]
    
    def add_delta_prob(self, ngram, delta=1):
        if ngram[:-1] not in self.ngram_freq:
            return 1 / len(self.vocab)
        counter = self.ngram_freq[ngram[:-1]]
        return (counter[ngram[-1]] + delta) / (self.n_1_gram_freq[ngram[:-1]] + delta*len(self.vocab))
    
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
        return 2 ** (total_log_prob / total_words)
