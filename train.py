# external libraries
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# internal utilities
import config
from model import BiDAF
from data_loader import SquadDataset
from utils import custom_sampler, save_checkpoint, exact_match

# hyper-parameters setup
hyper_params = {
    "num_epochs": config.num_epochs,
    "batch_size": config.batch_size,
    "valid_size": config.valid_size,
    "learning_rate": config.learning_rate,
    "cuda": config.cuda,
    "pretrained": config.pretrained
}

device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(42)

# define a path to save experiment logs
experiment_path = "output/{}".format(config.exp)
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

# start TensorboardX writer
writer = SummaryWriter(experiment_path)

# open features file and store them in individual variables
features = np.load(os.path.join(config.train_dir, "train_features.npz"))
w_context = features["context_idxs"]
c_context = features["context_char_idxs"]
w_question = features["question_idxs"]
c_question = features["question_char_idxs"]
labels = features["label"]

# load the embedding matrix created for our word vocabulary
with open(os.path.join(config.train_dir, "word_embeddings.pkl"), "rb") as e:
    word_embedding_matrix = pickle.load(e)
with open(os.path.join(config.train_dir, "char_embeddings.pkl"), "rb") as e:
    char_embedding_matrix = pickle.load(e)

# transform them into Tensors
word_embedding_matrix = torch.from_numpy(np.array(word_embedding_matrix)).type(torch.float32)
char_embedding_matrix = torch.from_numpy(np.array(char_embedding_matrix)).type(torch.float32)

print("Creating dataset...")
part_a_dataset_train = SquadDataset(w_context, c_context, w_question, c_question, labels)
part_a_dataset_valid = SquadDataset(w_context, c_context, w_question, c_question, labels)
print("Dataset sucessfully loaded!")

# define a split for train/valid
train_sampler, valid_sampler = custom_sampler(data=w_context, valid_size=hyper_params["valid_size"])

# load data generators
print("Loading dataloader...")
train_dataloader = DataLoader(part_a_dataset_train,
                              shuffle=False,
                              batch_size=hyper_params["batch_size"],
                              sampler=train_sampler, num_workers=4)

valid_dataloader = DataLoader(part_a_dataset_valid,
                              shuffle=False,
                              batch_size=hyper_params["batch_size"],
                              sampler=valid_sampler)

print("Dataloader sucessfully loaded!")

print("Length of training data loader is:", len(train_dataloader))
print("Length of valid data loader is:", len(valid_dataloader))

print("Loading model...")

model = BiDAF(word_vectors=word_embedding_matrix,
              char_vectors=char_embedding_matrix,
              hidden_size=config.hidden_size,
              drop_prob=config.drop_prob)
model.to(device)

print("Model successfully loaded!")

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), hyper_params["learning_rate"], weight_decay=1e-4)

# best loss so far
if hyper_params["pretrained"]:
    best_valid_loss = torch.load(os.path.join(experiment_path, "model.pkl"))["best_valid_loss"]
    epoch_checkpoint = torch.load(os.path.join(experiment_path, "model_last_checkpoint.pkl"))["epoch"]
else:
    best_valid_loss = 100
    epoch_checkpoint = 0

print("Best validation loss so far is:", best_valid_loss)

# train the Model
for epoch in range(hyper_params["num_epochs"]):
    print("##### epoch {:2d}".format(epoch))
    model.train()
    train_losses = 0
    train_ems = 0
    for i, batch in enumerate(train_dataloader):
        w_context, c_context, w_question, c_question, label1, label2 = batch[0].long().to(device),\
                                                                       batch[1].long().to(device), \
                                                                       batch[2].long().to(device), \
                                                                       batch[3].long().to(device), \
                                                                       batch[4][:, 0].long().to(device),\
                                                                       batch[4][:, 1].long().to(device)
        optimizer.zero_grad()
        pred1, pred2 = model(w_context, c_context, w_question, c_question)
        loss = criterion(pred1, label1) + criterion(pred2, label2)
        train_losses += loss.item()
        train_ems += exact_match(pred1, pred2, label1, label2)

        loss.backward()
        optimizer.step()

#        if (i + 1) % 1 == 0:
#            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f"
#                  % (epoch + 1, hyper_params["num_epochs"], i + 1, len(train_dataloader), loss.item()))
#            print("Number of exact matches in batch:", exact_match(pred1, pred2, label1, label2))

    writer.add_scalars("train", {"loss": np.round(train_losses / len(train_dataloader), 2),
                       "EM": np.round(train_ems / len(train_dataloader), 2),
                       "epoch": epoch + 1})
    print("Train loss of the model at epoch {} is: {}".format(epoch + 1, np.round(train_losses /
                                                                                  len(train_dataloader), 2)))
    print("Train EM of the model at epoch {} is: {}".format(epoch + 1, np.round(train_ems /
                                                                                len(train_dataloader), 2)))

    model.eval()
    valid_losses = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            w_context, c_context, w_question, c_question, label1, label2 = batch[0].long().to(device), \
                                                batch[1].long().to(device), \
                                                batch[2].long().to(device), \
                                                batch[3].long().to(device), \
                                                batch[4][:, 0].long().to(device), \
                                                batch[4][:, 1].long().to(device)
            pred1, pred2 = model(w_context, c_context, w_question, c_question)
            loss = criterion(pred1, label1) + criterion(pred2, label2)
            valid_losses += loss.item()

        writer.add_scalars("valid", {"loss": np.round(valid_losses / len(valid_dataloader), 2),
                          "epoch": epoch + 1})
        print("Validation loss of the model at epoch {} is: {}".format(epoch + 1, np.round(valid_losses /
                                                                                           len(valid_dataloader), 2)))

    # save last model weights
    save_checkpoint({
        "epoch": epoch + 1 + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": np.round(valid_losses / len(valid_dataloader), 2)
    }, True, os.path.join(experiment_path, "model_last_checkpoint.pkl"))

    # save model with best validation error
    is_best = bool(np.round(valid_losses / len(valid_dataloader), 2) < best_valid_loss)
    best_valid_loss = min(np.round(valid_losses / len(valid_dataloader), 2), best_valid_loss)
    save_checkpoint({
        "epoch": epoch + 1 + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": best_valid_loss
    }, is_best, os.path.join(experiment_path, "model.pkl"))

# export scalar data to JSON for external processing
writer.export_scalars_to_json(os.path.join(experiment_path, "all_scalars.json"))
writer.close()
