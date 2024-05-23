import pandas as pd
import numpy as np
import statsmodels.api as sm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 在此我指定使用2号GPU，可根据需要调整
import torch
import gc
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertModel, BertForPreTraining
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset

os.makedirs("/kaggle/working/models", exist_ok=True)
class Config:
    TRAIN_DATA_PATH = r'/kaggle/input/imdb-movie-review-sentiment-dataset'
    TEST_DATA_PATH = r'/kaggle/input/imdb-movie-review-sentiment-dataset'

    # Path to the directory where the model will be saved
    MODEL_DIR = '/kaggle/working/models'

def load_data(tokenizer):
    # Create a list of documents
    # print(train_set.head())
    # print(test_set.head())
    
    data_files = {"train": "train.csv", "test": "test.csv"}
    train_dataset = load_dataset(path=Config.TRAIN_DATA_PATH, split='train', data_files=data_files)
    test_dataset = load_dataset(path=Config.TEST_DATA_PATH, split='test', data_files=data_files)
    # dataset_structure: [text, label]

    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    MAX_LENGTH = 512
    train_dataset = train_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)


    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

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

model = AutoModelForSequenceClassification.from_config(model_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    learning_rate=3e-4,
    num_train_epochs=3,              # total number of training epochs
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

def get_accuracy(model, train_dataset, test_dataset, device='cuda:0'):
    model=model.to(device)
    BATCH_SIZE = 32
    X_train = []
    X_test = []
    
    with torch.no_grad():
        for i in range(0, len(train_dataset), BATCH_SIZE):
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

    train_labels = train_dataset['labels']
    test_labels = test_dataset['labels']

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


# test_acc = get_accuracy_pipeline(classifier, test_dataset, tokenizer)
# print(f"Test accuracy: {test_acc}")
test_acc = get_accuracy(model, train_dataset, test_dataset)

