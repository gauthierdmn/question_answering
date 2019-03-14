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
from config import data_dir
from utils import tokenizer, clean_text, word_tokenize, build_vocab

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

    def convert_idx(self, text, tokens):
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print("Token {} cannot be found".format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans

    def split_data(self, filename):
        self.load_data(filename)
        sub_dir = filename.split('-')[0]

        # create a subdirectory for Train and Dev data
        if not os.path.exists(os.path.join(self.data_dir, sub_dir)):
            os.makedirs(os.path.join(self.data_dir, sub_dir))

        with open(os.path.join(self.data_dir, sub_dir, sub_dir + '.context'), 'w', encoding='utf-8') as context_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '.question'), 'w', encoding='utf-8') as question_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '.answer'), 'w', encoding='utf-8') as answer_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '.labels'), 'w', encoding='utf-8') as labels_file:

            # loop over the data
            for article_id in tqdm.tqdm(range(len(self.data['data']))):
                list_paragraphs = self.data['data'][article_id]['paragraphs']
                # loop over the paragraphs
                for paragraph in list_paragraphs:
                    context = paragraph['context']
                    context = clean_text(context)
                    context_tokens = word_tokenize(context)
                    # ?? context_chars = [list(token) for token in context_tokens]
                    spans = self.convert_idx(context, context_tokens)
                    qas = paragraph['qas']
                    # loop over Q/A
                    for qa in qas:
                        question = qa['question']
                        question = clean_text(question)
                        question_tokens = word_tokenize(question)
                        # select only one ground truth, the top answer
                        answer_id = 0
                        try:
                            answer = qa['answers'][answer_id]['text']
                            answer = clean_text(answer)
                            answer_tokens = word_tokenize(answer)
                            answer_start = qa['answers'][answer_id]['answer_start']
                            answer_stop = answer_start + len(answer)
                        except:
                            # question does not have answer (SQuAD 2.0 specificity, not handled here)
                            continue
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_stop <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]

                        # write to file
                        context_file.write(' '.join([token for token in context_tokens]) + '\n')
                        question_file.write(' '.join([token for token in question_tokens]) + '\n')
                        answer_file.write(' '.join([token for token in answer_tokens]) + '\n')
                        labels_file.write(str(y1) + ' ' + str(y2) + '\n')

    def preprocess(self):
        self.split_data(train_filename)
        self.split_data(dev_filename)

    def extract_features(self, max_len=100, is_train=True):
        # choose the right directory
        directory = 'train' if is_train else 'dev'

        # download vocabulary if not done yet
        build_vocab(directory + '.context', directory + '.question', 'vocab.pkl', 'word2idx.pkl', is_train=is_train)

        # load word2idx dictionary
        with open(os.path.join(self.data_dir, directory, 'word2idx.pkl'), 'rb') as w:
            word2idx = pickle.load(w)
        # load context
        with open(os.path.join(self.data_dir, directory, directory + '.context'), 'r') as c:
            context = c.readlines()
        # load questions
        with open(os.path.join(self.data_dir, directory, directory + '.question'), 'r') as q:
            question = q.readlines()
        # load answer
        with open(os.path.join(self.data_dir, directory, directory + '.answer'), 'r') as a:
            answer = a.readlines()

        # clean and tokenize context and question
        context = [[w.lower().strip('\n') for w in word_tokenize(clean_text(doc))] for doc in context]
        question = [[w.lower().strip('\n') for w in word_tokenize(clean_text(doc))] for doc in question]

        print("Number of context paragraphs before filtering:", len(context))
        filter = [len(c) < max_len for c in context]
        context, question, answer = zip(*[(np.array(c), np.array(q), np.array(a)) for c, q, a, f in zip(context, question, answer, filter) if f])
        print("Number of context paragraphs after filtering ", len(context))
        print("Answer:", answer[0])

        # replace the tokenized words with their associated ID in the vocabulary
        context_features = []
        question_features = []
        for i, (c, q) in tqdm.tqdm(enumerate(zip(context, question))):
            # create empty numpy arrays
            context_idxs = np.zeros([max_len], dtype=np.int32)
            question_idxs = np.zeros([max_len], dtype=np.int32)
            # replace 0 values with word IDs
            for i, w in enumerate(c):
                context_idxs[i] = word2idx[w]
            for i, w in enumerate(q):
                question_idxs[i] = word2idx[w]
            # put the features in the features list
            context_features.append(context_idxs)
            question_features.append(question_idxs)

        # save context as numpy arrays
        with open(os.path.join(self.data_dir, directory, directory + '_context.pkl'), 'wb') as c:
            pickle.dump(context_features, c)
        # save question as numpy arrays
        with open(os.path.join(self.data_dir, directory, directory + '_question.pkl'), 'wb') as q:
            pickle.dump(question_features, q)
        # save answer as numpy arrays
        with open(os.path.join(self.data_dir, directory, directory + '_answer.pkl'), 'wb') as a:
            pickle.dump(answer, a)


if __name__ == "__main__":
    train_filename = "train-v2.0.json"
    dev_filename = "dev-v2.0.json"

    maybe_download_squad(url, train_filename, data_dir)
    maybe_download_squad(url, dev_filename, data_dir)

    p = SquadPreprocessor(data_dir, train_filename, dev_filename, tokenizer)
    #p.preprocess()

    p.extract_features(max_len=100, is_train=True)
