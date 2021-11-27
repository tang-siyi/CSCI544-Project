import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch import nn
from bigru import BiGRU, train, predict

GLOVE_MODEL_PATH = '../dataset/glove.6B.100d'
CLEAN_FINE_DATA_DIR = '../dataset/fine_data_clean.csv'
LOAD_PRETRAINED_MODEL = False


def generate_glove_vocab_embeddings(glove_path):
    with open(glove_path, 'rt', encoding='utf8') as f:
        full_content = f.read().strip().split('\n')

    glove_dict, embeddings = {'<pad>': 0, '<eos>': 1, '<unk>': 2}, []
    for line in full_content:
        vals = line.split(' ')
        word, embedding = vals[0], [float(x) for x in vals[1:]]
        if word not in glove_dict:
            glove_dict[word] = len(glove_dict)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    
    # embeddings for special tokens: <pad>, <eos>, <unk>
    pad_embedding = np.zeros((1, embeddings.shape[1]))
    eos_embedding = np.ones((1, embeddings.shape[1]))
    unk_embedding = np.mean(embeddings, axis=0, keepdims=True)
    
    #insert embeddings for pad and unk tokens at top of embeddings.
    embeddings = np.vstack((pad_embedding, eos_embedding, unk_embedding, embeddings))

    return glove_dict, embeddings


def get_word_index(word_dict, word):
    if word in word_dict:
        return word_dict[word]
    # return <unk> if not in the training dictionary
    return 2


def convert_words(corpus, word_dict):
    poems = []
    for poem in corpus:
        poem_vec = [get_word_index(word_dict, x) for x in str(poem).split(' ')]
        poem_vec.append(1)
        poems.append(poem_vec)

    return poems

def evaluate_score(y_true, y_pred):
    print('micro f1: %.6f\n' % f1_score(y_true, y_pred, average='micro'),
          'macro f1: %.6f\n' % f1_score(y_true, y_pred, average='macro'),
          'weighted f1: %.6f\n' % f1_score(y_true, y_pred, average='weighted'))


if __name__ == '__main__':
    raw_data = pd.read_csv(CLEAN_FINE_DATA_DIR).sample(frac=1.0, random_state=19).reset_index(drop=True)
    poems, labels = raw_data['poem'].to_numpy(), raw_data['label'].to_numpy(dtype=int)
    label_num = len(np.unique(labels))
    #print(poems, len(poems))
    #print(labels, len(labels))

    glove_dict, embeddings = generate_glove_vocab_embeddings(GLOVE_MODEL_PATH)
    #print(glove_dict)


    X = convert_words(poems, glove_dict)
    #print(X)
    #print(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=23)
    #print(X_train, y_train)

    model = BiGRU(label_num)
    model.init_embedding(embeddings)

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()
    # specify optimizer and learning rate
    learning_rate = 8e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, )
    print('learning_rate =', learning_rate)

    # train RNN
    n_epochs = 25
    if not LOAD_PRETRAINED_MODEL:
        bigru = train(model, 'bigru.pt',
                       X_train, y_train,
                       criterion, optimizer, n_epochs=n_epochs)
    else:
        bigru = model
        bigru.load_state_dict(torch.load('bigru.pt'))

    bigru.eval()

    y_pred = predict(bigru, X_train)
    print('----------- evalution on train set ----------')
    evaluate_score(y_train, y_pred)


    y_pred = predict(bigru, X_test)

    # for i in range(len(y_pred)):
    #     if y_test[i] != y_pred[i].item():
    #         print(y_test[i], y_pred[i])
    print('----------- evalution on test set ----------')
    evaluate_score(y_test, y_pred)

