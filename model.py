# external libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Text(nn.Module):

    def __init__(self, features, embedding_matrix, output_dim):
        super(CNN_Text, self).__init__()

        self.embedding_matrix = torch.from_numpy(embedding_matrix)
        V = embedding_matrix.shape[0]
        D = embedding_matrix.shape[1]

        self.embed = nn.Embedding(V, D, padding_idx=0)
        self.embed.load_state_dict({'weight': self.embedding_matrix})
        self.embed.weight.requires_grad = False

        self.conv = nn.Conv1d(50, 100, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(features, output_dim)

        # Freeze weights
        #for p in self.features.parameters():
        #    p.required_grad = False

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, context, question):
        context = self.embed(context)
        context = context.permute(0, 2, 1)

        context = self.conv(context)

        context = context.view(context.size(0), -1)
        context = self.dropout(context)

        out = self.fc1(context)

        return out
