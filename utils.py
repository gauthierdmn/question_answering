# external libraries
import os
import pickle
import numpy as np
from collections import Counter
from spacy.lang.en import English
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn

# internal utilities
import config

tokenizer = English()
device = torch.device("cuda" if config.cuda else "cpu")


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("''", '" ').replace("``", '" ')

    return text


def word_tokenize(sent):
    return [token.text for token in tokenizer(sent)]


def build_vocab(context_filename, question_filename, word_vocab_filename, word2idx_filename,
                char_vocab_filename, char2idx_filename, is_train=True, max_words=-1):
    # select the directory we want to create the vocabulary from
    directory = config.train_dir if is_train else config.dev_dir

    # load the context and question files
    with open(os.path.join(directory, context_filename), 'r', encoding="utf-8") as context,\
         open(os.path.join(directory, question_filename), 'r', encoding="utf-8") as question:
        context_file = context.readlines()
        question_file = question.readlines()

    # clean and tokenize the texts
    words = [w.lower() for doc in context_file + question_file for w in word_tokenize(clean_text(doc))]
    chars = [c for w in words for c in list(w)]
    # create a dictionary with word and char frequencies
    word_vocab = Counter(words)
    char_vocab = Counter(chars)
    # put them in a list ordered by frequency
    word_vocab = ['--NULL--'] + ['--UNK--'] + sorted(word_vocab, key=word_vocab.get, reverse=True)
    char_vocab = ['--NULL--'] + ['--UNK--'] + sorted(char_vocab, key=char_vocab.get, reverse=True)
    # limit the word vocabulary to top max_words
    word_vocab = word_vocab[:max_words]
    # get the word and char to ID dictionary mapping
    word2idx = dict([(x, y) for (y, x) in enumerate(word_vocab)])
    char2idx = dict([(x, y) for (y, x) in enumerate(char_vocab)])

    # save those files
    with open(os.path.join(directory, word_vocab_filename), 'wb') as wv, \
         open(os.path.join(directory, word2idx_filename), 'wb') as wd, \
        open(os.path.join(directory, char_vocab_filename), 'wb') as cv, \
        open(os.path.join(directory, char2idx_filename), 'wb') as cd:
        pickle.dump(word_vocab, wv)
        pickle.dump(word2idx, wd)
        pickle.dump(char_vocab, cv)
        pickle.dump(char2idx, cd)

    print("Vocabulary created successfully.")
    return word_vocab, word2idx, char_vocab, char2idx


def build_embeddings(vocab, embedding_path="", output_path="", vec_size=50):
    embedding_dict = {}
    # Load pretrained embeddings if an embedding path is provided
    if embedding_path:
        # Get the path associated to the embedding size we want
        embedding_path = embedding_path.format(vec_size)
        with open(embedding_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                if word in vocab:
                    embedding_dict[word] = vector

    embedding_dict['--NULL--'] = np.asarray([0. for _ in range(vec_size)])
    embedding_dict['--UNK--'] = np.asarray([0. for _ in range(vec_size)])
    embedding_matrix = []
    count = 0
    for v in vocab:
        if v in embedding_dict:
            embedding_matrix.append(embedding_dict[v])
        else:
            count += 1
            embedding_matrix.append(np.random.normal(0, 0.1, vec_size))
    # Save the embedding matrix
    with open(os.path.join(config.train_dir, output_path), "wb") as e:
        pickle.dump(embedding_matrix, e)


def save_checkpoint(state, is_best, filename="/output/checkpoint.pkl"):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best model.")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation loss did not improve.")


def custom_sampler(data, valid_size=0.02):
    # Define a split for train/valid
    num_train = len(data)

    indices = list(range(num_train))
    split = int(np.floor((1 - valid_size) * num_train))

    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs


def to_ids(pred1, pred2):
    batch_size, c_len = pred1.size()
    ls = nn.LogSoftmax(dim=1)
    mask = (torch.ones(c_len, c_len) * float("-inf")).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
    score = (ls(pred1).unsqueeze(2) + ls(pred2).unsqueeze(1)) + mask
    score, s_idx = score.max(dim=1)
    score, e_idx = score.max(dim=1)
    s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

    return s_idx, e_idx


def exact_match(p1, p2, l1, l2):
    p1, p2 = to_ids(p1, p2)
    if device == "cpu":
        return sum([l1.numpy()[i] == p1.numpy()[i] and l2.numpy()[i] == p2.numpy()[i] for i in range(len(l1))])
    else:
        return sum([l1.cpu().numpy()[i] == p1.cpu().numpy()[i] and l2.cpu().numpy()[i] == p2.cpu().numpy()[i]
                    for i in range(len(l1))])
