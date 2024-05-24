import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 在此我指定使用2号GPU，可根据需要调整
# ban wandb
os.environ["WANDB_DISABLED"] = "true"
import torch
import gc

from sentence_transformers import util
from sentence_transformers import SentenceTransformer

os.makedirs("/kaggle/working/models", exist_ok=True)
class Config:
    TRAIN_DATA_PATH = r'/kaggle/input/imdb-movie-review-sentiment-dataset/train.csv'
    TEST_DATA_PATH = r'/kaggle/input/imdb-movie-review-sentiment-dataset/test.csv'

    # Path to the directory where the model will be saved
    MODEL_DIR = '/kaggle/working/models'

def check_dataset(train_dataset, test_dataset):
    print(train_dataset)
    print(train_dataset[0])
    print(test_dataset)
    print(test_dataset[0])

embed_model = SentenceTransformer("/kaggle/input/nomic-embed-text-v1-5-model/nomic-embed-text-v1.5", trust_remote_code=True)
def load_data():
    # Create a list of documents
    train_set = pd.read_csv(Config.TRAIN_DATA_PATH)
    test_set = pd.read_csv(Config.TEST_DATA_PATH)
    # print(train_set.head())
    # print(test_set.head())
    # dataset_structure: [text, label]

    return train_set, test_set


def encode_in_batches(my_model, documents, batch_size=8):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = list(documents[i:i+batch_size])
            batch_embeddings = my_model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            embeddings.extend(batch_embeddings.detach().cpu().numpy())  

            gc.collect()
            torch.cuda.empty_cache()
    embeddings = np.array(embeddings)
    return embeddings

def nomic_embed_encode_in_batches(documents, batch_size=8):
    from nomic import embed

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = list(documents[i:i+batch_size])
            batch_embeddings = embed.text(
                texts=batch,
                model='nomic-embed-text-v1.5',
                task_type='search_document',
                inference_mode = 'local',
                device = 'gpu',
                dimensionality=512,
            )
            embeddings.extend(batch_embeddings.detach().cpu().numpy())  

            gc.collect()
            torch.cuda.empty_cache()
    embeddings = np.array(embeddings)
    return embeddings

    


def test_nomic(model):
    train_set, test_set = load_data()
    # check_dataset(train_set, test_set)

    IS_VECTOR_SAVED = False
    if not IS_VECTOR_SAVED:
        X_train = encode_in_batches(model, train_set["text"].values)
        np.save(Config.MODEL_DIR+"/train_vectors_nomic.npy", X_train)
        X_test = encode_in_batches(model, test_set["text"].values)
        np.save(Config.MODEL_DIR+"/test_vectors_nomic.npy", X_test)
        # X_train = nomic_embed_encode_in_batches(train_set["text"])
        # X_test = nomic_embed_encode_in_batches(test_set["text"])
    
    X_train = np.load(Config.MODEL_DIR+"/train_vectors_nomic.npy")
    X_test = np.load(Config.MODEL_DIR+"/test_vectors_nomic.npy")

    print("Embeddings are ready.")
    print("Number of documents in training set: ", len(X_train))
    print("Number of documents in test set: ", len(X_test))
    
    train_labels = train_set['label'].values
    test_labels = test_set['label'].values

    print("Start training the Logistic Regression model.")
    clf = sm.Logit(train_labels, X_train).fit()
    print("Logistic Regression Model is trained.")
    # print(clf.summary())

    # Evaluate the model
    train_prediction = np.where(clf.predict(X_train) > 0.5, 1, 0)
    test_prediction = np.where(clf.predict(X_test) > 0.5, 1, 0)

    train_acc = np.mean(train_prediction == train_labels)
    test_acc = np.mean(test_prediction == test_labels)

    print("Train accuracy: ", train_acc)
    print("Test accuracy: ", test_acc)

    return test_acc

gc.collect()
test_nomic(embed_model)
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