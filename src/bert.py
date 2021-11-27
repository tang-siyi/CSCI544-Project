import pandas as pd
pd.set_option('max_colwidth',150)
pd.options.mode.chained_assignment = None
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup


CLEAN_FINE_DATA_DIR = '../dataset/fine_data_clean.csv'
DATASET_DIR = '../dataset/'
FINE_DATA_DIR = DATASET_DIR+'fine_data.csv'
BATCH_SIZE=4
LOG_MODE=True

def load_data(data_dir):
    raw_data = pd.read_csv(data_dir)[:20]
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
        #print(poem_str)
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


def encode(poems):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    embeddings = []
    for poem in poems:
        embedding = tokenizer.encode(poem, add_special_tokens = True,
                                     padding = True, truncation=True, return_tensors = 'pt')
        embeddings.append(embedding[0])
    return embeddings


def train(model, model_dir, X, y, optimizer, n_epochs=5):
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
            # clear the gradients of all optimized variables
            model.zero_grad()

            X_encoded = encode(X)
            X_lens = [x.shape[0] for x in X_encoded]
            X_padded = pad_sequence(X_encoded, batch_first=True)

            output = model(X_padded,
                           token_type_ids=None,
                           attention_mask=(X_padded>0),
                           labels=torch.tensor(target, dtype=torch.float))

            loss = output['loss']
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            scheduler.step()
            # update running training loss
            train_loss += loss.item()

        # validate the model
        model.eval() # prep model for evaluation
        for X, target in zip(X_valid_batch, y_valid_batch):
            X_encoded = encode(X)
            X_lens = [x.shape[0] for x in X_encoded]
            X_padded = pad_sequence(X_encoded, batch_first=True)

            output = model(X_padded,
                           token_type_ids=None,
                           attention_mask=(X_padded>0),
                           labels=torch.tensor(target, dtype=torch.float))

            loss = output['loss']
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            scheduler.step()
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
    preds = []

    model.eval() # prep model for evaluation
    with torch.no_grad():
        for X in X_batch:
            X_encoded = encode(X)
            X_lens = [x.shape[0] for x in X_encoded]
            X_padded = pad_sequence(X_encoded, batch_first=True)

            output = model(X_padded,
                           token_type_ids=None,
                           attention_mask=(X_padded>0),
                           labels=torch.tensor(target, dtype=torch.float))
            logits = outputs['logits']
            #logits = logits.detach().cpu().numpy()
            preds.append(np.argmax(logits))

    return preds


if __name__ == '__main__':
    #raw_data = pd.read_csv(CLEAN_FINE_DATA_DIR).sample(frac=1.0, random_state=19).reset_index(drop=True)[:20]
    #poems, labels = raw_data['poem'].to_numpy(), raw_data['label'].to_numpy(dtype=int)
    poems, labels = load_data(FINE_DATA_DIR)
    label_num = len(np.unique(labels))

    X_train, X_test, y_train, y_test = train_test_split(poems, labels, test_size=0.2, random_state=23)
    
    n_epochs = 10

    # Load the pretrained BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                    num_labels=label_num, output_attentions=False, output_hidden_states=False)
    
    # create optimizer and learning rate schedule
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train(model, 'bert.pt', X_train, y_train, optimizer, n_epochs)
    y_pred = predict(model, X_test)
