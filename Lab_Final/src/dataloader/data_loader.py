import pandas as pd

basic_prompt_fmt = r'''
SENTIMENT ANALYSIS TASK:
Carefully examine the movie review provided below and ascertain the overall sentiment expressed in the review. You should classify the sentiment as either positive or negative. Please enclose your conclusion within the brackets, like so: [answer].

REVIEW:
'''

basic_prompt_fmt_2 = r'''Please perform Sentiment Classification Task. Given the Sentence from imdb below, assign a sentiment label from ['negative','positive']. Return label only without any other text.\n Here is the Sentence: '''


CARP_prompt_fmt = r'''
This is an overall sentiment classifier for movie reviews.
First, list CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, 
references) that support the sentiment determination of input..
Second, deduce the diagnostic REASONING process from premises (i.e., clues, input) that supports the INPUT 
sentiment determination (Limit the number of words to 130).
Third, based on clues, reasoning and input, determine the overall SENTIMENT of INPUT as Positive or Negative. You should classify the sentiment as either positive or negative.

For example:
INPUT: press the delete key
CLUES: delete key
REASONING: The phrase "delete key" implies an action of removing something, which could be interpreted as a negative sentiment.
SENTIMENT: Negative

INPUT: He successfully managed to create a heartwarming story that resonated with the audience.
CLUES: successfully, heartwarming story, resonated
REASONING: The use of "successfully" indicates accomplishment, "heartwarming story" suggests positive emotions, "resonated" implies a strong positive connection, and mentioning it was with the "audience" highlights broad appeal, all pointing towards a favorable view, which could be interpreted as a positive sentiment.
SENTIMENT: Positive

Please provide your response in the following format:
CLUES: [clues]
REASONING: [reasoning]
SENTIMENT: [sentiment]

Now it's your turn. Please analyze the sentiment of the movie review below:
INPUT: '''

def CARP_prompt(problem):
    """
    @misc{sun2023text,
      title={Text Classification via Large Language Models}, 
      author={Xiaofei Sun and Xiaoya Li and Jiwei Li and Fei Wu and Shangwei Guo and Tianwei Zhang and Guoyin Wang},
      year={2023},
      eprint={2305.08377},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
    }
    """

    return CARP_prompt_fmt+problem

class TestDataLoader:
    def __init__(self, data_path):
        self.test_data_path = data_path + "/plain_text/test-00000-of-00001.parquet"
        self.test_set = pd.read_parquet(self.test_data_path)[::10]
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
        review, label = self.load_data(idx)
        # print(review, label)
        return basic_prompt_fmt_2 + review, label
    
    def CARP_prompt(self, idx):
        review, label = self.load_data(idx)
        return CARP_prompt(review), label
