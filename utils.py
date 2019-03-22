# external libraries
import os
import spacy
import pickle
import tqdm
import numpy as np
from collections import Counter
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import torch
from torch.utils.data.sampler import SubsetRandomSampler

# internal utilities
from config import train_dir, dev_dir, spacy_en

nlp = spacy.load(spacy_en)
# tokenizer = Tokenizer(nlp.vocab)


def custom_en_tokenizer(en_vocab):
     prefixes = list(English.Defaults.prefixes)
     prefixes.remove('>')
     prefix_re = spacy.util.compile_prefix_regex(tuple(prefixes))

     suffixes = list(English.Defaults.suffixes)
     suffixes.remove('>')
     suffixes.remove('<')
     suffix_re = spacy.util.compile_suffix_regex(tuple(suffixes))

     infixes = list(English.Defaults.infixes)
     infixes.append('>')
     infixes.append('<')
     infix_re = spacy.util.compile_infix_regex(tuple(infixes))

     return Tokenizer(en_vocab,
                      English.Defaults.tokenizer_exceptions,
                      prefix_re.search,
                      suffix_re.search,
                      infix_re.finditer,
                      token_match=None)

tokenizer = custom_en_tokenizer(nlp.vocab)

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("''", '" ').replace("``", '" ')

    return text


def word_tokenize(sent):
    return [token.text for token in tokenizer(sent)]


def build_vocab(context_filename, question_filename, vocab_filename, word2idx_filename, is_train=True, max_words=-1):
    # select the directory we want to create the vocabulary from
    directory = train_dir if is_train else dev_dir

    # load the context and question files
    with open(os.path.join(directory, context_filename), 'r', encoding="utf-8") as context,\
         open(os.path.join(directory, question_filename), 'r', encoding="utf-8") as question:
        context_file = context.readlines()
        question_file = question.readlines()

    # clean and tokenize the texts
    words = [w.lower() for doc in context_file + question_file for w in word_tokenize(clean_text(doc))]
    # create a dictionary with word frequencies
    vocab = Counter(words)
    # put them in a list ordered by frequency
    vocab = ['--NULL--'] + ['--UNK--'] + sorted(vocab, key=vocab.get, reverse=True)
    # limit the vocabulary to top max_words
    vocab = vocab[:max_words]
    # get the word to ID dictionary mapping
    word2idx = dict([(x, y) for (y, x) in enumerate(vocab)])

    # save those files
    with open(os.path.join(directory, vocab_filename), 'wb') as v, \
         open(os.path.join(directory, word2idx_filename), 'wb') as d:
        pickle.dump(vocab, v)
        pickle.dump(word2idx, d)

    print("Vocabulary created successfully.")
    return vocab, word2idx


def build_word_embeddings(vocab, embedding_path="", vec_size=50):
    # Get the path associated to the embedding size we want
    embedding_path = embedding_path.format(vec_size)
    embedding_dict = {}
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
    print("Did not find {} words in GloVe out of {} words.".format(count, len(vocab)))
    # Save the embedding matrix
    with open(os.path.join(train_dir, "embeddings.pkl"), "wb") as e:
        pickle.dump(embedding_matrix, e)


def save_checkpoint(state, is_best, filename='/output/checkpoint.pkl'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best model.")
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
