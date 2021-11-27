import numpy as np
import pandas as pd
import time
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from sklearn.model_selection import train_test_split


BATCH_SIZE = 32
LOG_MODE = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu"')


""" NN model for classification """
"""
    NN structure: GloVe Embedding -> Bi-GRU -> Multi-head Attention
"""
class BiGRU(nn.Module):
    def __init__(self, output_num):
        super(BiGRU, self).__init__()

        self.embedding_dim = 100
        self.hidden_dim = 256
        self.output_dim = 128
        self.layer_num = 1
        self.bi_num = 2

        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim,
                          bidirectional=True, batch_first=True)

        self.attn_fc = nn.Linear(self.hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

        # p=0.33: 0.59(f1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.elu = nn.ELU(alpha=1.0)

        self.classifier = nn.Linear(self.output_dim, output_num)

        self.init_weight()


    def forward(self, input):
        self.batch_size = len(input)
        input_lens = [len(x) for x in input]

        embeddings = [self.embedding(torch.tensor(x).to(device)) for x in input]
        embedding_padded = pad_sequence(embeddings, batch_first=True)
        embedding_padded = self.dropout(embedding_padded)
        embedding_len = embedding_padded.shape[1]
        embedding_packed = pack(embedding_padded, input_lens,\
                                batch_first=True, enforce_sorted=False)

        # bi-rnn
        hidden_0 = self.init_hidden(embedding_len)
        output, _ = self.rnn(embedding_packed, hidden_0)
        (output, output_len) = unpack(output, batch_first=True)

        hiddens = output.chunk(2, dim=-1)
        H = torch.mul(hiddens[0], hiddens[1])

        # attention
        # ref: Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
        M = torch.tanh(H)
        alpha = self.softmax(self.attn_fc(M))
        r = torch.mul(H, alpha)
        output = torch.tanh(r)

        output = self.dropout(output)
        output = self.linear(output)
        output = self.elu(output)
        output = self.classifier(output)
        return (output, output_len)


    def init_weight(self):
        nn.init.kaiming_uniform_(self.rnn.weight_ih_l0.data)
        nn.init.kaiming_uniform_(self.rnn.weight_hh_l0.data)
        nn.init.xavier_uniform_(self.attn_fc.weight.data)
        nn.init.xavier_uniform_(self.linear.weight.data)
        nn.init.xavier_uniform_(self.classifier.weight.data)

        nn.init.constant_(self.rnn.bias_ih_l0.data, 0) 
        nn.init.constant_(self.rnn.bias_hh_l0.data, 0) 
        nn.init.constant_(self.attn_fc.bias.data, 0)
        nn.init.constant_(self.linear.bias.data, 0) 
        nn.init.constant_(self.classifier.bias.data, 0) 
    
    def init_hidden(self, input_dim):
        return torch.zeros(self.layer_num*self.bi_num, self.batch_size, self.hidden_dim).to(device)
   
    # init embedding with GloVe
    def init_embedding(self, embeddings):
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())

    
    def generate_mask(self, lens):
        max_len = np.max(lens)
        mask = torch.zeros(len(lens), max_len)
        for i in range(len(lens)):
            mask[i][:lens[i]] = 1
        return mask


def batch_data(X):
    X_batch = []
    for start in range(0, len(X), BATCH_SIZE):
        X_batch.append(X[start : start+BATCH_SIZE])
    return X_batch



def train(model, model_dir, X, y, criterion, optimizer, n_epochs=5):
    model = model.to(device)

    # prepare input data
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X, y, test_size=0.2, random_state=9)

    # batch data
    X_train_batch, y_train_batch = batch_data(X_train), batch_data(y_train)
    X_valid_batch, y_valid_batch = batch_data(X_valid), batch_data(y_valid)

    #print(X_train_batch)

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf # set initial "min" to infinity

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    for epoch in range(n_epochs):
        start = time.process_time()
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
    
        # train the model
        model.train() # prep model for training
        for X, target in zip(X_train_batch, y_train_batch):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            output, output_len = model(X)
            N = output.size(0)
            predictions = torch.stack([output[i][output_len[i]-1].float() for i in range(N)]).to(device)
            targets = torch.LongTensor(target).to(device)

            # calculate the loss
            loss = criterion(predictions, targets) / N
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            #scheduler.step()
            # update running training loss
            train_loss += loss.item()*torch.max(output_len).item()

        # validate the model
        model.eval() # prep model for evaluation
        with torch.no_grad():
            for X, target in zip(X_valid_batch, y_valid_batch):
                # forward pass: compute predicted outputs by passing inputs to the model
                output, output_len = model(X)
                N = output.size(0)
                predictions = torch.stack([output[i][output_len[i]-1].float() for i in range(N)]).to(device)
                targets = torch.LongTensor(target).to(device)

                # calculate the loss
                loss = criterion(predictions, targets) / N
                # update running validation loss 
                valid_loss += loss.item()*torch.max(output_len).item()

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

    y_pred = []
    with torch.no_grad():
        for batch_idx in range(len(X_batch)):
            # forward pass: compute predicted outputs by passing inputs to the model
            output, output_len = model(X_batch[batch_idx])
            for sentence_idx in range(len(output)):
                prediction = output[sentence_idx][output_len[sentence_idx]-1].float()
                y_pred.append(torch.argmax(softmax(prediction), dim=0).cpu())

    return y_pred

