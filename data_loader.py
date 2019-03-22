# external libraries
import torch.utils.data as data


class SquadDataset(data.Dataset):
    """Custom Dataset for SQuAD data compatible with torch.utils.data.DataLoader."""

    def __init__(self, context, question, labels):
        """Set the path for audio data, together wth labels and objid."""
        self.context = context
        self.question = question
        self.labels = labels

    def __getitem__(self, index):
        """Returns one data trio (context, question, answer)."""
        return self.context[index], self.question[index], self.labels[index]

    def __len__(self):
        return len(self.context)
