# external libraries
import os
import tqdm
import json
import zipfile
import tarfile
import pickle
import numpy as np
import urllib.request

# internal utilities
import config
from utils import tokenizer, clean_text, word_tokenize, build_vocab, build_embeddings, convert_idx

# URL to download SQuAD dataset 2.0
url = "https://rajpurkar.github.io/SQuAD-explorer/dataset"


def maybe_download_squad(url, filename, out_dir):
    # path for local file.
    save_path = os.path.join(out_dir, filename)

    # check if the file already exists
    if not os.path.exists(save_path):
        # check if the output directory exists, otherwise create it.
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print("Downloading", filename, "...")

        # download the dataset
        url = os.path.join(url, filename)
        file_path, _ = urllib.request.urlretrieve(url=url, filename=save_path)

    print("File downloaded successfully!")

    if filename.endswith(".zip"):
        # unpack the zip-file.
        print("Extracting ZIP file...")
        zipfile.ZipFile(file=filename, mode="r").extractall(out_dir)
        print("File extracted successfully!")
    elif filename.endswith((".tar.gz", ".tgz")):
        # unpack the tar-ball.
        print("Extracting TAR file...")
        tarfile.open(name=filename, mode="r:gz").extractall(out_dir)
        print("File extracted successfully!")


class SquadPreprocessor:
    def __init__(self, data_dir, train_filename, dev_filename, tokenizer):
        self.data_dir = data_dir
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.data = None
        self.tokenizer = tokenizer

    def load_data(self, filename="train-v2.0.json"):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath) as f:
            self.data = json.load(f)

    def split_data(self, filename):
        self.load_data(filename)
        sub_dir = filename.split('-')[0]

        # create a subdirectory for Train and Dev data
        if not os.path.exists(os.path.join(self.data_dir, sub_dir)):
            os.makedirs(os.path.join(self.data_dir, sub_dir))

        with open(os.path.join(self.data_dir, sub_dir, sub_dir + '.context'), 'w', encoding="utf-8") as context_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '.question'), 'w', encoding="utf-8") as question_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '.answer'), 'w', encoding="utf-8") as answer_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '.labels'), 'w', encoding="utf-8") as labels_file:

            # loop over the data
            for article_id in tqdm.tqdm(range(len(self.data['data']))):
                list_paragraphs = self.data['data'][article_id]['paragraphs']
                # loop over the paragraphs
                for paragraph in list_paragraphs:
                    context = paragraph['context']
                    context = clean_text(context)
                    context_tokens = [w for w in word_tokenize(context) if w]
                    spans = convert_idx(context, context_tokens)
                    qas = paragraph['qas']
                    # loop over Q/A
                    for qa in qas:
                        question = qa['question']
                        question = clean_text(question)
                        question_tokens = [w for w in word_tokenize(question) if w]
                        if sub_dir == "train":
                            # select only one ground truth, the top answer, if any answer
                            answer_ids = 1 if qa['answers'] else 0
                        else:
                            answer_ids = len(qa['answers'])
                        labels = []
                        if answer_ids:
                            for answer_id in range(answer_ids):
                                answer = qa['answers'][answer_id]['text']
                                answer = clean_text(answer)
                                answer_tokens = [w for w in word_tokenize(answer) if w]
                                answer_start = qa['answers'][answer_id]['answer_start']
                                answer_stop = answer_start + len(answer)
                                answer_span = []
                                for idx, span in enumerate(spans):
                                    if not (answer_stop <= span[0] or answer_start >= span[1]):
                                        answer_span.append(idx)
                                if not answer_span:
                                    continue
                                labels.append(str(answer_span[0]) + ' ' + str(answer_span[-1]))

                            # write to file
                            context_file.write(' '.join([token for token in context_tokens]) + '\n')
                            question_file.write(' '.join([token for token in question_tokens]) + '\n')
                            answer_file.write(' '.join([token for token in answer_tokens]) + '\n')
                            labels_file.write("|".join(labels) + "\n")

    def preprocess(self):
        self.split_data(train_filename)
        self.split_data(dev_filename)

    def extract_features(self, max_len_context=config.max_len_context, max_len_question=config.max_len_question,
                         max_len_word=config.max_len_word, is_train=True):
        # choose the right directory
        directory = "train" if is_train else "dev"

        # load context
        with open(os.path.join(self.data_dir, directory, directory + ".context"), "r", encoding="utf-8") as c:
            context = c.readlines()
        # load questions
        with open(os.path.join(self.data_dir, directory, directory + ".question"), "r", encoding="utf-8") as q:
            question = q.readlines()
        # load answer
        with open(os.path.join(self.data_dir, directory, directory + ".labels"), "r", encoding="utf-8") as l:
            labels = l.readlines()

        # clean and tokenize context and question
        context = [[w for w in word_tokenize(clean_text(doc.strip('\n')))] for doc in context]
        question = [[w for w in word_tokenize(clean_text(doc.strip('\n')))] for doc in question]

        # download vocabulary if not done yet
        if directory == "train":
            labels = [np.array(l.strip("\n").split(), dtype=np.int32) for l in labels]

            word_vocab, word2idx, char_vocab, char2idx = build_vocab(directory + ".context", directory + ".question",
                                                                     "word_vocab.pkl", "word2idx.pkl", "char_vocab.pkl",
                                                                     "char2idx.pkl", is_train=is_train,
                                                                     max_words=config.max_words)
            # create an embedding matrix from the vocabulary with pretrained vectors (GloVe) for words
            build_embeddings(word_vocab, embedding_path=config.glove, output_path="word_embeddings.pkl",
                             vec_size=config.word_embedding_size)
            build_embeddings(char_vocab, embedding_path="", output_path="char_embeddings.pkl",
                             vec_size=config.char_embedding_size)

        else:
            labels = np.array([l.strip("\n") for l in labels])

            with open(os.path.join(self.data_dir, "train", "word2idx.pkl"), "rb") as wi,\
                 open(os.path.join(self.data_dir, "train", "char2idx.pkl"), "rb") as ci:
                    word2idx = pickle.load(wi)
                    char2idx = pickle.load(ci)

        print("Number of questions before filtering:", len(question))
        filter = [len(c) < max_len_context and max([len(w) for w in c]) < max_len_word and
                  len(q) < max_len_question and max([len(w) for w in q]) < max_len_word and
                  len(q) > 3 for c, q in zip(context, question)]
        context, question, labels = zip(*[(c, q, l) for c, q, l, f in zip(
                                          context, question, labels, filter) if f])
        print("Number of questions after filtering ", len(question))

        # replace the tokenized words with their associated ID in the vocabulary
        context_idxs = []
        context_char_idxs = []
        question_idxs = []
        question_char_idxs = []
        for i, (c, q) in tqdm.tqdm(enumerate(zip(context, question))):
            # create empty numpy arrays
            context_idx = np.zeros([max_len_context], dtype=np.int32)
            question_idx = np.zeros([max_len_question], dtype=np.int32)
            context_char_idx = np.zeros([max_len_context, max_len_word], dtype=np.int32)
            question_char_idx = np.zeros([max_len_question, max_len_word], dtype=np.int32)

            # replace 0 values with word and char IDs
            for j, word in enumerate(c):
                if word in word2idx:
                    context_idx[j] = word2idx[word]
                else:
                    context_idx[j] = 1
                for k, char in enumerate(word):
                    if char in char2idx:
                        context_char_idx[j, k] = char2idx[char]
                    else:
                        context_char_idx[j, k] = 1
            context_idxs.append(context_idx)
            context_char_idxs.append(context_char_idx)

            for j, word in enumerate(q):
                if word in word2idx:
                    question_idx[j] = word2idx[word]
                else:
                    question_idx[j] = 1
                for k, char in enumerate(word):
                    if char in char2idx:
                        question_char_idx[j, k] = char2idx[char]
                    else:
                        question_char_idx[j, k] = 1
            question_idxs.append(question_idx)
            question_char_idxs.append(question_char_idx)
        # save features as numpy arrays
        np.savez(os.path.join(self.data_dir, directory, directory + "_features"),
                 context_idxs=np.array(context_idxs),
                 context_char_idxs=np.array(context_char_idxs),
                 question_idxs=np.array(question_idxs),
                 question_char_idxs=np.array(question_char_idxs),
                 label=np.array(labels))


if __name__ == "__main__":
    train_filename = "train-v2.0.json"
    dev_filename = "dev-v2.0.json"

    maybe_download_squad(url, train_filename, config.data_dir)
    maybe_download_squad(url, dev_filename, config.data_dir)

    p = SquadPreprocessor(config.data_dir, train_filename, dev_filename, tokenizer)
    p.preprocess()

    p.extract_features(max_len_context=config.max_len_context, max_len_question=config.max_len_question,
                       max_len_word=config.max_len_word, is_train=True)
    p.extract_features(max_len_context=config.max_len_context, max_len_question=config.max_len_question,
                       max_len_word=config.max_len_word, is_train=False)
