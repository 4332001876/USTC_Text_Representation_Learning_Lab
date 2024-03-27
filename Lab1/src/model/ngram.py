from dataloader import NgramDataset
from collections import Counter
import random

class NgramModel:
    def __init__(self, n, ngram_dataset: NgramDataset):
        assert n > 1
        self.vocab = self.ngram_dataset.vocab
        self.n = n
        self.ngram_dataset = ngram_dataset
        self.ngram_dataset.set_n(n)
        self.ngram_freq = self.build_ngram()

    def build_ngram(self):
        ngram_freq = {}
        for idx in range(len(self.ngram_dataset)):
            data = self.ngram_dataset[idx]
            for ngram in data:
                if ngram[:-1] not in ngram_freq:
                    ngram_freq[ngram[:-1]] = [ngram[-1]]
                else:
                    ngram_freq[ngram[:-1]].append(ngram[-1])

        for k, v in ngram_freq.items():
            ngram_freq[k] = Counter(v)
        return ngram_freq

    def generate(self, start, max_len=10):
        n = self.n
        res = start
        for _ in range(max_len):
            if start not in self.ngram_freq:
                break
            next_word = self.get_next_word(start) # random.choice(self.ngram_freq[tuple(start)])
            res += next_word
            start = res[-n+1:]
        return res
    
    def get_next_word(self, start):
        # return random.choice(self.ngram_freq[tuple(start)])
        # 根据概率选择下一个词
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
