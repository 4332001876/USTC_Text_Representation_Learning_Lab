import pandas as pd
import numpy as np
import lightgbm as lgb
import statsmodels.api as sm

# from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from config import Config

REGRESSION_METHOD = 'lgbm'  # 'lgbm' or 'logistic'

lgb_params = { 
    'verbose': -1, 
    'subsample_freq': 1, 
    'subsample': 0.95, 
    'skip_drop': 0.1, 
    'n_estimators': 2000, 
    'min_data_in_leaf': 10,
    'min_sum_hessian_in_leaf': 1, 
    'min_child_weight': 0.001,
    'min_gain_to_split': 0.2, 
    'max_drop': 30, 
    'max_depth': 5, 
    'max_bin': 32, 
    'learning_rate': 0.05, 
    'drop_seed': 123, 
    'drop_rate': 0.05, 
    'colsample_bytree': 0.8, 
    'reg_alpha': 0.06,
    'reg_lambda': 0.06,
    'objective': 'binary',
    'n_jobs': 8
}

# 文档分词
def tokenizing(doc):
    norm_doc = doc.lower()
    # Replace breaks with spaces
    norm_doc = norm_doc.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_doc = norm_doc.replace(char, ' ' + char + ' ')

    words = norm_doc.split()
    return words

def load_data():
    # Create a list of documents
    train_set = pd.read_parquet(Config.TRAIN_DATA_PATH)
    test_set = pd.read_parquet(Config.TEST_DATA_PATH)
    # print(dataset.head())
    # dataset_structure: [text, label]

    train_documents = [TaggedDocument(tokenizing(doc), [i]) for i, doc in enumerate(train_set['text'].values)]
    train_labels = train_set['label'].values
    test_documents = [TaggedDocument(tokenizing(doc), [i]) for i, doc in enumerate(test_set['text'].values)]
    test_labels = test_set['label'].values
    # print(documents[0:5])
    print("Documents are ready.")
    print("Number of documents in training set: ", len(train_documents))
    return train_documents, train_labels, test_documents, test_labels

def train_doc2vec():
    train_documents, train_labels, test_documents, test_labels = load_data()

    # 使用下面四种算法训练 doc2vec 向量： (1) HS + PV-DM:(2) HS + PV-DBOW:(3) NS + PV-DM:(4) NS + PV-DBOW
    hs_desc = {0: "Negative sampling", 1: "Hierarchical softmax"}
    dm_desc = {0: "PV-DBOW", 1: "PV-DM"}
    hs_names = {0: "NS", 1: "HS"}
    dm_names = {0: "DBOW", 1: "DM"}
    for hs in [0, 1]:
        for dm in [0, 1]:
            print("Training model with hs={}, dm={}".format(hs_names[hs], dm_names[dm]))
            model = Doc2Vec(train_documents, vector_size=200, window=5, min_count=5, workers=8, hs=hs, dm=dm, dm_concat=0, dm_mean=1, epochs=20)
            print("Model is ready.")

            # Save the model
            model.save(Config.MODEL_DIR+"doc2vec_model_hs={}_dm={}.model".format(hs_names[hs], dm_names[dm]))

            # Infer a vector for a new document
            # vector = model.infer_vector(["system", "response"])
            # print(vector)

            # Train a LGBM classifier
            X_train = np.array([model.infer_vector(doc.words) for doc in train_documents])
            X_test = np.array([model.infer_vector(doc.words) for doc in test_documents])

            # save these vectors
            np.save(Config.MODEL_DIR+"train_vectors_hs={}_dm={}.npy".format(hs_names[hs], dm_names[dm]), X_train)
            np.save(Config.MODEL_DIR+"test_vectors_hs={}_dm={}.npy".format(hs_names[hs], dm_names[dm]), X_test)

            
            #  情感分析任务方法：逻辑回归
            if REGRESSION_METHOD == 'logistic':
                clf = sm.Logit(train_labels, X_train).fit()
                print("Logistic Regression Model is trained.")
                print(clf.summary())

                # Evaluate the model
                train_acc = clf.predict(X_train)
                test_acc = clf.predict(X_test)

            elif REGRESSION_METHOD == 'lgbm':
                clf = lgb.LGBMClassifier(**lgb_params)
                clf.fit(X_train, train_labels)
                print("LGBMClassifier Model is trained.")

                # Evaluate the model
                train_acc = clf.score(X_train, train_labels)
                test_acc = clf.score(X_test, test_labels)

            print("Train accuracy: ", train_acc)
            print("Test accuracy: ", test_acc)
            print("--------------------------------------------------")

def test_doc2vec():
    IS_VECTOR_SAVED = True
    train_documents, train_labels, test_documents, test_labels = load_data()

    # 使用下面四种算法训练 doc2vec 向量： (1) HS + PV-DM:(2) HS + PV-DBOW:(3) NS + PV-DM:(4) NS + PV-DBOW
    hs_desc = {0: "Negative sampling", 1: "Hierarchical softmax"}
    dm_desc = {0: "PV-DBOW", 1: "PV-DM"}
    hs_names = {0: "NS", 1: "HS"}
    dm_names = {0: "DBOW", 1: "DM"}
    for hs in [0, 1]:
        for dm in [0, 1]:
            print("Testing model with hs={}, dm={}".format(hs_names[hs], dm_names[dm]))
            if not IS_VECTOR_SAVED:
                model = Doc2Vec.load(Config.MODEL_DIR+"doc2vec_model_hs={}_dm={}.model".format(hs_names[hs], dm_names[dm]))

                # Train a LGBM classifier
                X_train = np.array([model.infer_vector(doc.words) for doc in train_documents])
                X_test = np.array([model.infer_vector(doc.words) for doc in test_documents])

                # save these vectors
                np.save(Config.MODEL_DIR+"train_vectors_hs={}_dm={}.npy".format(hs_names[hs], dm_names[dm]), X_train)
                np.save(Config.MODEL_DIR+"test_vectors_hs={}_dm={}.npy".format(hs_names[hs], dm_names[dm]), X_test)

            # load these vectors
            X_train = np.load(Config.MODEL_DIR+"train_vectors_hs={}_dm={}.npy".format(hs_names[hs], dm_names[dm]))
            X_test = np.load(Config.MODEL_DIR+"test_vectors_hs={}_dm={}.npy".format(hs_names[hs], dm_names[dm]))

            #  情感分析任务方法：逻辑回归
            if REGRESSION_METHOD == 'logistic':
                clf = sm.Logit(train_labels, X_train).fit()
                print("Logistic Regression Model is trained.")
                print(clf.summary())

                # Evaluate the model
                train_acc = clf.predict(X_train)
                test_acc = clf.predict(X_test)
                
            elif REGRESSION_METHOD == 'lgbm':
                # clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.08, min_child_samples=75, n_jobs=8)
                clf = lgb.LGBMClassifier(**lgb_params)
                clf.fit(X_train, train_labels)
                print("LGBMClassifier Model is trained.")

                # Evaluate the model
                train_acc = clf.score(X_train, train_labels)
                test_acc = clf.score(X_test, test_labels)

            print("Train accuracy: ", train_acc)
            print("Test accuracy: ", test_acc)
            print("--------------------------------------------------")


if __name__ == '__main__':
    # train_doc2vec()
    test_doc2vec()
