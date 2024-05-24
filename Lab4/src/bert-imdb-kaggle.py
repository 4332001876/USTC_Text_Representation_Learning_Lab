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
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertModel, BertForMaskedLM, AutoModelForCausalLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset

os.makedirs("/kaggle/working/models", exist_ok=True)
class Config:
    TRAIN_DATA_PATH = r'/kaggle/input/imdb-movie-review-sentiment-dataset'
    TEST_DATA_PATH = r'/kaggle/input/imdb-movie-review-sentiment-dataset'

    MLM_DATA_PATH = '/kaggle/working/mlm_data.txt'

    # Path to the directory where the model will be saved
    MODEL_DIR = '/kaggle/working/models'

def check_dataset(train_dataset, test_dataset):
    print(train_dataset)
    print(train_dataset[0])
    print(test_dataset)
    print(test_dataset[0])

def load_data(tokenizer):
    data_files = {"train": "train.csv", "test": "test.csv"}
    train_dataset = load_dataset(path=Config.TRAIN_DATA_PATH, split='train', data_files=data_files)
    test_dataset = load_dataset(path=Config.TEST_DATA_PATH, split='test', data_files=data_files)
    # dataset_structure: [text, label]

    # check_dataset(train_dataset, test_dataset)

    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    MAX_LENGTH = 512
    train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

    # check_dataset(train_dataset, test_dataset)

    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    # check_dataset(train_dataset, test_dataset)

    return train_dataset, test_dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

model_id = 'prajjwal1/bert-tiny'
# note that we need to specify the number of classes for this task
# we can directly use the metadata (num_classes) stored in the dataset
model_config = AutoConfig.from_pretrained(model_id)
# we can also modify the configuration of the model
HIDDEN_SIZE = 100
model_config.hidden_size = HIDDEN_SIZE
model_config.intermediate_size = 4 * HIDDEN_SIZE
model_config.num_attention_heads = 4
# model_config.num_hidden_layers = 2

# classifier("love")
def get_accuracy_pipeline(classifier, test_dataset, tokenizer):
    test_dataset_text = [tensor[:480] for tensor in test_dataset['input_ids']]
    # token_id to text
    test_dataset_text = tokenizer.batch_decode(test_dataset_text)
    predictions = [classifier(test_dataset_text[i]) for i in range(len(test_dataset))]  
    labels = [x['labels'] for x in test_dataset]
    preds = [int(label[0]['label'][-1]) for label in predictions]
    acc = accuracy_score(labels, preds)
    return acc

def get_accuracy(model, train_dataset, test_dataset, device='cuda:0', is_classifier=True):
    model=model.to(device)
    BATCH_SIZE = 1024
    X_train = []
    X_test = []
    
    with torch.no_grad():
        # for i in range(0, len(train_dataset), BATCH_SIZE): # to_tqdm
        for i in tqdm(range(0, len(train_dataset), BATCH_SIZE)):
            train_dataset_outputs = model(
                train_dataset['input_ids'][i:i+BATCH_SIZE].to(device),
                train_dataset['token_type_ids'][i:i+BATCH_SIZE].to(device),
                train_dataset['attention_mask'][i:i+BATCH_SIZE].to(device),
            )
            # print(i)
            # print(train_dataset_outputs)
            X_train.append(train_dataset_outputs.logits.cpu().detach().numpy())  # 使用pooler_output作为标签的向量表示

            test_dataset_outputs = model(
                test_dataset['input_ids'][i:i+BATCH_SIZE].to(device),
                test_dataset['token_type_ids'][i:i+BATCH_SIZE].to(device),
                test_dataset['attention_mask'][i:i+BATCH_SIZE].to(device),
            )
            X_test.append(test_dataset_outputs.logits.cpu().detach().numpy())  # 使用pooler_output作为标签的向量表示

            gc.collect()
            torch.cuda.empty_cache()
    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)

    train_labels = train_dataset['labels'].numpy()
    test_labels = test_dataset['labels'].numpy()

    if not is_classifier:
        print("Start training the Logistic Regression model.")
        clf = sm.Logit(train_labels, X_train).fit()
        print("Logistic Regression Model is trained.")
        # print(clf.summary())

        # Evaluate the model
        train_prediction = np.where(clf.predict(X_train) > 0.5, 1, 0)
        test_prediction = np.where(clf.predict(X_test) > 0.5, 1, 0)

        train_acc = np.mean(train_prediction == train_labels)
        test_acc = np.mean(test_prediction == test_labels)
    else:

        train_acc = np.mean(np.argmax(X_train, axis=1) == train_labels)
        test_acc = np.mean(np.argmax(X_test, axis=1) == test_labels)

    print("Train accuracy: ", train_acc)
    print("Test accuracy: ", test_acc)

    return test_acc

def get_accuracy_mlm(model, train_dataset, test_dataset, device='cuda:0'):
    model=model.to(device)
    BATCH_SIZE = 512

    IS_VECTOR_SAVED = False
    if not IS_VECTOR_SAVED:
        X_train = []
        X_test = []
        
        with torch.no_grad():
            # for i in range(0, len(train_dataset), BATCH_SIZE): # to_tqdm
            for i in tqdm(range(0, len(train_dataset), BATCH_SIZE)):
                train_dataset_outputs = model(
                    train_dataset['input_ids'][i:i+BATCH_SIZE].to(device),
                    train_dataset['token_type_ids'][i:i+BATCH_SIZE].to(device),
                    train_dataset['attention_mask'][i:i+BATCH_SIZE].to(device),
                )
                # print(i)
                # print(train_dataset_outputs)
                # print(train_dataset_outputs.last_hidden_state[:,0,:].shape)
                # X_train.append(train_dataset_outputs.last_hidden_state[:,0,:].cpu().detach().numpy())  # 使用pooler_output作为标签的向量表示
                X_train.append(torch.mean(train_dataset_outputs.last_hidden_state,dim=1).cpu().detach().numpy())

                test_dataset_outputs = model(
                    test_dataset['input_ids'][i:i+BATCH_SIZE].to(device),
                    test_dataset['token_type_ids'][i:i+BATCH_SIZE].to(device),
                    test_dataset['attention_mask'][i:i+BATCH_SIZE].to(device),
                )
                # X_test.append(test_dataset_outputs.last_hidden_state[:,0,:].cpu().detach().numpy())  # 使用pooler_output作为标签的向量表示
                X_test.append(torch.mean(test_dataset_outputs.last_hidden_state,dim=1).cpu().detach().numpy()) 

                gc.collect()
                torch.cuda.empty_cache()
        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)

        # save these vectors
        np.save(Config.MODEL_DIR+"train_vectors_mlm.npy", X_train)
        np.save(Config.MODEL_DIR+"test_vectors_mlm.npy", X_test)

    train_labels = train_dataset['labels'].numpy()
    test_labels = test_dataset['labels'].numpy()

    # load these vectors
    X_train = np.load(Config.MODEL_DIR+"train_vectors_mlm.npy")
    X_test = np.load(Config.MODEL_DIR+"test_vectors_mlm.npy")

    gc.collect()

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
    

def load_data_mlm(tokenizer):
    # merge unsupervised and train_data:
    unsupervised_data = pd.read_csv(Config.TRAIN_DATA_PATH + '/unsupervised.csv')
    train_data = pd.read_csv(Config.TRAIN_DATA_PATH + '/train.csv')
    merged_data = pd.concat([unsupervised_data['text'], train_data['text']], ignore_index=True)
    # output to a file
    merged_data.to_csv(Config.MLM_DATA_PATH, index=False, header=False)

    # gen LineByLineTextDataset
    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=Config.MLM_DATA_PATH, block_size=128)
    test_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=Config.TEST_DATA_PATH + '/test.csv', block_size=128)
    return train_dataset, test_dataset


def train_mlm():
    model = BertForMaskedLM(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        learning_rate=3e-4,
        num_train_epochs=10,              # total number of training epochs
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.1,
        weight_decay=0.01,
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        do_train=True,
        do_eval=True,
        no_cuda=False,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    train_dataset, test_dataset = load_data_mlm(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    trainer = Trainer(
        model=model,                         # the instantiated   Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,            # evaluation dataset
        data_collator=data_collator,
    )

    train_out = trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")

    # save the model
    model.save_pretrained(Config.MODEL_DIR)

train_mlm()

# model = AutoModelForSequenceClassification.from_pretrained(model_id, config=model_config)
# model = AutoModelForSequenceClassification.from_config(model_config)
model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(model_id)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    learning_rate=3e-4,
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
    do_train=True,
    do_eval=True,
    no_cuda=False,
    load_best_model_at_end=True,
    # eval_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch"
    
)

train_dataset, test_dataset = load_data(tokenizer)

trainer = Trainer(
    model=model,                         # the instantiated   Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)

train_out = trainer.train()

# save the model
model.save_pretrained(Config.MODEL_DIR)

# load the model
model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_DIR)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

gc.collect()
torch.cuda.empty_cache()


# test_acc = get_accuracy_pipeline(classifier, test_dataset, tokenizer)
# print(f"Test accuracy: {test_acc}")
test_acc = get_accuracy(model, train_dataset, test_dataset)

#test mlm
# model = BertModel.from_pretrained(Config.MODEL_DIR)
# model = BertModel.from_pretrained(model_id)
# get_accuracy_mlm(model, train_dataset, test_dataset)