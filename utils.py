# external libraries
import os
import spacy
import pickle
from collections import Counter
from spacy.tokenizer import Tokenizer

# internal utilities
from config import train_dir, dev_dir

nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)


def clean_text(text):
	text = text.replace("\n", " ")
	text = text.replace("''", '" ').replace("``", '" ')

	return text


def word_tokenize(sent):
	return [token.text for token in tokenizer(sent)]


def build_vocab(context_filename, question_filename, vocab_filename, word2idx_filename, is_train=True):
	# select the directory we want to create the vocabulary from
	directory = train_dir if is_train else dev_dir

	# test whether or not the vocab was already created before
	if not os.path.exists(os.path.join(directory, vocab_filename)) and not os.path.exists(os.path.join(directory, word2idx_filename)):
		# load the context and question files
		with open(os.path.join(directory, context_filename), 'r', encoding="utf-8") as context, \
			open(os.path.join(directory, question_filename), 'r', encoding="utf-8") as question:
			context_file = context.readlines()
			question_file = question.readlines()

		# clean and tokenize the texts
		words = [w.lower() for doc in context_file + question_file for w in word_tokenize(clean_text(doc))]
		# create a dictionary with word frequencies
		vocab = Counter(words)
		# put them in a list ordered by frequency
		vocab = ['--NULL--'] + ['--UNK--'] + sorted(vocab, key=vocab.get, reverse=True)
		# get the word to ID dictionary mapping
		word2idx = dict([(x, y) for (y, x) in enumerate(vocab)])

		# save those files
		with open(os.path.join(directory, vocab_filename), 'wb') as v, \
			open(os.path.join(directory, word2idx_filename), 'wb') as d:
			pickle.dump(vocab, v)
			pickle.dump(word2idx, d)

	print("Vocabulary created successfully.")
