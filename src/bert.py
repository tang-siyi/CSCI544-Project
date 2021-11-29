import pandas as pd
pd.set_option('max_colwidth',150)
pd.options.mode.chained_assignment = None
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import BertModel
from transformers import get_linear_schedule_with_warmup


CLEAN_FINE_DATA_DIR = '../dataset/fine_data_clean.csv'
DATASET_DIR = '../dataset/'
FINE_DATA_DIR = DATASET_DIR+'fine_data.csv'


BATCH_SIZE = 1
LOG_MODE = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data(data_dir):
    raw_data = pd.read_csv(data_dir)
    #print(raw_data.head(3))
    
    data_pt = raw_data[["poem","our_tag"]]
    #print(data_pt.head(10))
    
    total_poem = len(data_pt["poem"])
    #print("total number poem: ", total_poem)
    poem_content = data_pt["poem"].tolist()
    #print(poem_content[1])
    
    # convert the poem content into a long string
    def reformat(raw_poem):
        poem = raw_poem[1:-1].replace("\"","").split(",")
        sentences = [x.strip(" ").strip("'") for x in poem]
        poem_str = ' '.join(sentences)
        return poem_str
    
    poems = [reformat(x) for x in poem_content]
    
    # convert label from string to integer
    labels = data_pt['our_tag'].value_counts().keys()
    label_map = {}
    for i in range(len(labels)):
        label_map[labels[i]] = i

    labels = np.array([label_map[x] for x in data_pt['our_tag']])

    return poems, labels


def batch_data(X):
    X_batch = []
    for start in range(0, len(X), BATCH_SIZE):
        X_batch.append(X[start : start+BATCH_SIZE])
    return X_batch




class BERT(nn.Module):
    def __init__(self, label_num):
        super(BERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                    num_labels=label_num, output_attentions=False, output_hidden_states=False)
    
    def forward(self, X):
        X_encoded = self.encode(X)
        X_padded = pad_sequence(X_encoded, batch_first=True)

        output = self.bert(X_padded, token_type_ids=None, attention_mask=(X_padded>0))
        #print('output: ', output)
        return output['logits']

    def encode(self, poems):
        embeddings = []
        for poem in poems:
            embedding = self.tokenizer.encode(poem, add_special_tokens = True, 
                            max_length=512, truncation = True, return_tensors = 'pt')
            #print(embedding[0].shape)
            embeddings.append(embedding[0].to(device))
        return embeddings



def train(model, model_dir, X, y, criterion, optimizer, n_epochs=5):
    model = model.to(device)

    # prepare input data
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X, y, test_size=0.2, random_state=9)

    # batch data
    X_train_batch, y_train_batch = batch_data(X_train), batch_data(y_train)
    X_valid_batch, y_valid_batch = batch_data(X_valid), batch_data(y_valid)

    valid_loss_min = np.Inf # set initial "min" to infinity

    total_steps = len(X_train_batch) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(n_epochs):
        start = time.process_time()
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
    
        # train the model
        model.train() # prep model for training
        for X, target in zip(X_train_batch, y_train_batch):
            start_batch = time.process_time()


            # clear the gradients of all optimized variables
            model.zero_grad()

            output = model(X)
            N = output.size(0)
            targets = torch.LongTensor(target).to(device)

            # calculate the loss
            loss = criterion(output, targets) / N
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            scheduler.step()
            # update running training loss
            train_loss += loss.item()

            end_batch = time.process_time()

            # if LOG_MODE:
            #     print('time each epoch: %.2f s' % (end_batch-start_batch))

        # validate the model
        model.eval() # prep model for evaluation
        with torch.no_grad():
            for X, target in zip(X_valid_batch, y_valid_batch):
                output = model(X)
                N = output.size(0)
                targets = torch.LongTensor(target).to(device)

                # calculate the loss
                loss = criterion(output, targets) / N
                # update running training loss
                valid_loss += loss.item()

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss / len(X_train_batch)
        valid_loss = valid_loss / len(X_valid_batch)


        if LOG_MODE:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch+1, 
                train_loss,
                valid_loss
                ))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            if LOG_MODE:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        valid_loss))
            torch.save(model.state_dict(), model_dir)
            valid_loss_min = valid_loss

        end = time.process_time()
        if LOG_MODE:
            print('time each epoch: %.2f s' % (end-start))

    model.load_state_dict(torch.load(model_dir))

    return model


def predict(model, X):
    X_batch = batch_data(X)
    softmax = nn.Softmax(dim=0)
    preds = []

    model = model.to(device)
    model.eval() # prep model for evaluation
    with torch.no_grad():
        for X in X_batch:
            output = model(X)
            for x in output:
                preds.append(torch.argmax(softmax(x), dim=0).cpu())

    return preds


def evaluate_score(y_true, y_pred):
    print('micro f1: %.6f\n' % f1_score(y_true, y_pred, average='micro'),
          'macro f1: %.6f\n' % f1_score(y_true, y_pred, average='macro'),
          'weighted f1: %.6f\n' % f1_score(y_true, y_pred, average='weighted'))



if __name__ == '__main__':
    poems, labels = load_data(FINE_DATA_DIR)
    label_num = len(np.unique(labels))

    X_train, X_test, y_train, y_test = train_test_split(poems, labels, test_size=0.2, random_state=23)
    
    n_epochs = 5

    # Load the pretrained BERT model
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
    #                num_labels=label_num, output_attentions=False, output_hidden_states=False)
    
    model = BERT(label_num)

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()
    # specify optimizer and learning rate
    learning_rate = 1e-6
    # create optimizer and learning rate schedule
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    LOAD_PRETRAINED_MODEL = True
    if not LOAD_PRETRAINED_MODEL:
        model = train(model, 'model_bert.pt', X_train, y_train, criterion, optimizer, n_epochs)
    else:
        model.load_state_dict(torch.load('model_bert.pt'))
        model.eval()
    
    y_pred = predict(model, X_test)
    # print(y_pred)
    print('----------- evalution on test set ----------')
    evaluate_score(y_test, y_pred)

