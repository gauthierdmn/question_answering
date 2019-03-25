# external libraries
import os
import re
import spacy
import pickle
import string
import numpy as np
from collections import Counter
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

# internal utilities
from config import train_dir, dev_dir, spacy_en

nlp = spacy.load("en_core_web_sm")
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


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.
    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.
    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs


# All methods below this line are from the official SQuAD 2.0 eval script
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1
