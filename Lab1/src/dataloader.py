from nltk.util import ngrams
from config import Config

class NgramDataset:
    def __init__(self, file_path, n=2):
        self.data = []
        with open(file_path, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                if len(words) == 0:
                    continue
                self.data.append(words)
        self.vocab = set([word for sent in self.data for word in sent])
        self.n = n

    def get_n(self):
        return self.n
    
    def set_n(self, n):
        self.n = n

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return list(ngrams(self.data[idx], self.n, pad_left=True, left_pad_symbol=Config.LEFT_PAD_SYMBOL, 
                           pad_right=True, right_pad_symbol=Config.RIGHT_PAD_SYMBOL))
    
    
    