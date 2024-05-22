import pandas as pd
import numpy as np
import statsmodels.api as sm

# os.makedirs("/kaggle/working/models")
class Config:
    TRAIN_DATA_PATH = r'/kaggle/input/imdb-movie-review-sentiment-dataset/train.csv'
    TEST_DATA_PATH = r'/kaggle/input/imdb-movie-review-sentiment-dataset/test.csv'

    # Path to the directory where the model will be saved
    MODEL_DIR = '/kaggle/working/'

def load_data():
    # Create a list of documents
    train_set = pd.read_csv(Config.TRAIN_DATA_PATH)
    test_set = pd.read_csv(Config.TEST_DATA_PATH)
    # print(train_set.head())
    # print(test_set.head())
    # dataset_structure: [text, label]

    train_documents = [TaggedDocument(tokenizing(doc), [i]) for i, doc in enumerate(train_set['text'].values)]
    train_labels = train_set['label'].values
    test_documents = [TaggedDocument(tokenizing(doc), [i]) for i, doc in enumerate(test_set['text'].values)]
    test_labels = test_set['label'].values
    # print(documents[0:5])
    print("Documents are ready.")
    print("Number of documents in training set: ", len(train_documents))
    return train_documents, train_labels, test_documents, test_labels


from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch

embed_model = SentenceTransformer("/kaggle/input/nomic-embed-text-v1-5-model/nomic-embed-text-v1.5", trust_remote_code=True)
search_q = f"search_query: {search_q_desc}"

def encode_in_batches(my_model, documents, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = list(documents[i:i+batch_size].values)
        batch_embeddings = my_model.encode(batch, convert_to_tensor=True)
        embeddings.extend(batch_embeddings)  

        torch.cuda.empty_cache()

    return embeddings

df["search_doc"] = df["desc"].apply(lambda x: "search_document: "+x)
doc_embeddings = encode_in_batches(embed_model, df["search_doc"])

'''
embeddings = embed_model.encode(sentences, convert_to_tensor=True)
embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
embeddings = embeddings[:, :matryoshka_dim]
embeddings = F.normalize(embeddings, p=2, dim=1)
similarities = util.cos_sim(q_embeddings[0], doc_embeddings)
similarities = similarities[0]

top_n = 50
top_k_values, top_k_indices = torch.topk(similarities, top_n)

top_k_rows = []
top_k_indices = top_k_indices[0]
for idx in top_k_indices:
    top_row = df.iloc[int(idx)]
    top_k_rows.append(top_row)
'''