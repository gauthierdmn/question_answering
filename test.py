# external libraries
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# internal utilities
import config
from model import BiDAF
from data_loader import SquadDataset
from utils import compute_batch_metrics

# preprocessing values used for training
prepro_params = {
    "max_words": config.max_words,
    "word_embedding_size": config.word_embedding_size,
    "char_embedding_size": config.char_embedding_size,
    "max_len_context": config.max_len_context,
    "max_len_question": config.max_len_question,
    "max_len_word": config.max_len_word
}

# hyper-parameters setup
hyper_params = {
    "num_epochs": config.num_epochs,
    "batch_size": config.batch_size,
    "learning_rate": config.learning_rate,
    "hidden_size": config.hidden_size,
    "char_channel_width": config.char_channel_width,
    "char_channel_size": config.char_channel_size,
    "drop_prob": config.drop_prob,
    "cuda": config.cuda,
    "pretrained": config.pretrained
}

experiment_params = {"preprocessing": prepro_params, "model": hyper_params}

# train on GPU if CUDA variable is set to True (a GPU with CUDA is needed to do so)
device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(42)

# define a path to save experiment logs
experiment_path = "output/{}".format(config.exp)
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

# start TensorboardX writer
writer = SummaryWriter(experiment_path)

# open features file and store them in individual variables
dev_features = np.load(os.path.join(config.dev_dir, "dev_features.npz"))
d_w_context, d_c_context, d_w_question, d_c_question, d_labels = dev_features["context_idxs"],\
                                                                 dev_features["context_char_idxs"],\
                                                                 dev_features["question_idxs"],\
                                                                 dev_features["question_char_idxs"],\
                                                                 dev_features["label"]

# load word2idx and idx2word dictionaries
with open(os.path.join(config.train_dir, "word2idx.pkl"), "rb") as f:
    word2idx = pickle.load(f)

idx2word = dict([(y, x) for x, y in word2idx.items()])

# load the embedding matrix created for our word vocabulary
with open(os.path.join(config.train_dir, "word_embeddings.pkl"), "rb") as e:
    word_embedding_matrix = pickle.load(e)
with open(os.path.join(config.train_dir, "char_embeddings.pkl"), "rb") as e:
    char_embedding_matrix = pickle.load(e)

# transform them into Tensors
word_embedding_matrix = torch.from_numpy(np.array(word_embedding_matrix)).type(torch.float32)
char_embedding_matrix = torch.from_numpy(np.array(char_embedding_matrix)).type(torch.float32)

# load dataset
test_dataset = SquadDataset(d_w_context, d_c_context, d_w_question, d_c_question, d_labels)

# load data generator
test_dataloader = DataLoader(test_dataset,
                             shuffle=True,
                             batch_size=hyper_params["batch_size"],
                             num_workers=4)

print("Length of test data loader is:", len(test_dataloader))

# load the model
model = BiDAF(word_vectors=word_embedding_matrix,
              char_vectors=char_embedding_matrix,
              hidden_size=hyper_params["hidden_size"],
              drop_prob=hyper_params["drop_prob"])
try:
    if config.cuda:
        model.load_state_dict(torch.load(os.path.join(config.squad_models, "model_final.pkl"))["state_dict"])
    else:
        model.load_state_dict(torch.load(os.path.join(config.squad_models, "model_final.pkl"),
                                         map_location=lambda storage, loc: storage)["state_dict"])
    print("Model weights successfully loaded.")
except:
    print("Model weights not found, initialized model with random weights.")
model.to(device)

# define loss criterion
criterion = nn.CrossEntropyLoss()

model.eval()
test_em = 0
test_f1 = 0
n_samples = 0
with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        w_context, c_context, w_question, c_question, labels = batch[0].long().to(device),\
                                                                     batch[1].long().to(device),\
                                                                     batch[2].long().to(device),\
                                                                     batch[3].long().to(device),\
                                                                     batch[4]
        pred1, pred2 = model(w_context, c_context, w_question, c_question)
        em, f1 = compute_batch_metrics(w_context, idx2word, pred1, pred2, labels)
        test_em += em
        test_f1 += f1
        n_samples += w_context.size(0)

    writer.add_scalars("test", {"EM": np.round(test_em / n_samples, 2),
                                "F1": np.round(test_f1 / n_samples, 2)})
    print("Test EM of the model after training is: {}".format(np.round(test_em / n_samples, 2)))
    print("Test F1 of the model after training is: {}".format(np.round(test_f1 / n_samples, 2)))

