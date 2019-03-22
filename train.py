# external libraries
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# internal utilities
import config
from model import CNN_Text
from data_loader import SquadDataset
from utils import custom_sampler, save_checkpoint

# hyper-parameters setup
hyper_params = {
    'num_epochs': config.num_epochs,
    'batch_size': config.batch_size,
    'valid_size': config.valid_size,
    'learning_rate': config.learning_rate,
    'output_dim': config.output_dim,
    'cuda': config.cuda,
    'pretrained': config.pretrained
}

device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(42)

# open labels file
with open(os.path.join(config.train_dir, "train_context.pkl"), "rb") as c:
    context = pickle.load(c)
# open labels file
with open(os.path.join(config.train_dir, "train_question.pkl"), "rb") as q:
    question = pickle.load(q)
# open labels file
with open(os.path.join(config.train_dir, "train_labels.pkl"), "rb") as l:
    labels = pickle.load(l)

# load the embedding matrix created for our vocabulary
with open(os.path.join(config.train_dir, "embeddings.pkl"), "rb") as e:
    embedding_matrix = pickle.load(e)

embedding_matrix = np.array(embedding_matrix)

print("Creating dataset...")
part_a_dataset_train = SquadDataset(context, question, labels)
part_a_dataset_valid = SquadDataset(context, question, labels)
print("Dataset sucessfully loaded!")

# define a split for train/valid
train_sampler, valid_sampler = custom_sampler(data=context, valid_size=hyper_params['valid_size'])

# load data generators
print("Loading dataloader...")
train_dataloader = DataLoader(part_a_dataset_train,
                        shuffle=False,
                        batch_size=hyper_params['batch_size'],
                        sampler=train_sampler, num_workers=4)

valid_dataloader = DataLoader(part_a_dataset_valid,
                        shuffle=False,
                        batch_size=hyper_params['batch_size'],
                        sampler=valid_sampler)

print("Dataloader sucessfully loaded!")

print("Length of training data loader is:", len(train_dataloader))
print("Length of valid data loader is:", len(valid_dataloader))

print("Loading model...")

model = CNN_Text(10000, embedding_matrix, hyper_params['output_dim'])
model.to(device)

print("Model successfully loaded!")

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=hyper_params['learning_rate'], momentum=0.9, weight_decay=1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])

# best loss so far
if hyper_params['pretrained']:
    best_valid_loss = torch.load('output/dummy.pkl')['best_valid_loss']
    epoch_checkpoint = torch.load('output/dummy._last_checkpoint.pkl')['epoch']
else:
    best_valid_loss = 100
    epoch_checkpoint = 0

print("Best validation loss so far is:", best_valid_loss)

# train the Model
for epoch in range(hyper_params['num_epochs']):
    print("##### epoch {:2d}".format(epoch))
    model.train()
    train_losses = 0
    for i, batch in enumerate(train_dataloader):
        context, question, label1, label2 = batch[0].long().to(device),\
                                            batch[1].long().to(device),\
                                            batch[2][:, 0].long().to(device),\
                                            batch[2][:, 1].long().to(device)
        optimizer.zero_grad()
        pred = model(context, question).squeeze(1)
        loss = criterion(pred, label1)
        train_losses += loss.item()

        loss.backward()
        optimizer.step()

        if (i + 1) % 400 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, hyper_params['num_epochs'], i + 1, len(train_dataloader), loss.item()))

    model.eval()
    valid_losses = 0
    n_samples = 0
    list_preds = []
    list_labels = []
    list_audio = []

    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            context, question, label1, label2 = batch[0].long().to(device), \
                                                batch[1].long().to(device), \
                                                batch[2][:, 0].long().to(device), \
                                                batch[2][:, 1].long().to(device)
            pred = model(context, question).squeeze(1)
            loss = criterion(pred, label1)
            valid_losses += loss.item()
            list_preds += [x for x in pred.data.numpy().tolist()]
            list_labels += label1.data.cpu().numpy().tolist()

        list_preds = [np.argmax(p) for p in list_preds]

        acc = round(100*sum([p == l for p, l in zip(list_preds, list_labels)]) / len(list_labels), 2)
        print('Validation accuracy of the model at epoch {} is: {} %'.format(epoch, acc))

    # save last model weights
    save_checkpoint({
        'epoch': epoch + 1 + epoch_checkpoint,
        'state_dict': model.state_dict(),
        'best_valid_loss': np.round(valid_losses / len(valid_dataloader), 2)
    }, True, 'output/dummy_last_checkpoint.pkl')

    # save model with best validation error
    is_best = bool(np.round(valid_losses / len(valid_dataloader), 2) < best_valid_loss)
    best_valid_loss = min(np.round(valid_losses / len(valid_dataloader), 2), best_valid_loss)
    save_checkpoint({
        'epoch': epoch + 1 + epoch_checkpoint,
        'state_dict': model.state_dict(),
        'best_valid_loss': best_valid_loss
    }, is_best, 'output/dummy.pkl'.format(hyper_params['learning_rate']))
