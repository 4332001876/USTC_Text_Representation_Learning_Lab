import pandas as pd

basic_prompt = r'''
SEMENTIC ANALYSIS:
Please read the movie review below and determine the sentiment of the review. The sentiment can be positive or negative. Please put the answer inside \boxed{}.
REVIEW:
{review}
'''

class TestDataLoader:
    def __init__(self, test_data_path):
        self.test_data_path = test_data_path
        self.test_set = pd.read_parquet(self.test_data_path)[0:100]
        # print(train_set.head())
        # print(test_set.head())
        # dataset_structure: [text, label]

        """
                                                        text  label
        0  I rented I AM CURIOUS-YELLOW from my video sto...      0
        1  "I Am Curious: Yellow" is a risible and preten...      0
        2  If only to avoid making this type of film in t...      0
        3  This film was probably inspired by Godard's Ma...      0
        4  Oh, brother...after hearing about this ridicul...      0

        label: 0:neg 1:pos
        """

        # to_numpy
        self.test_documents = self.test_set['text'].values
        self.test_labels = self.test_set['label'].values
        
    def load_data(self, idx):
        return self.test_documents[idx], self.test_labels[idx]
    
    def __len__(self):
        return len(self.test_documents)
    
    def __getitem__(self, idx):
        return self.load_data(idx)
    
    def basic_prompt(self, idx):
        document, label = self.load_data(idx)
        return basic_prompt.format(review=document), label
    
