import os
import torch
import torch.utils.data as data


class SquadDataset(data.Dataset):
    """Custom Dataset for SQuAD data compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, context_filename, question_filename, answer_filename):
        """Set the path for audio data, together wth labels and objid."""
        self.root = root
		self.context_filename = context_filename
		self.question_filename = question_filename
        self.answer_filename = answer_filename

    def __getitem__(self, index):
        """Returns one data trio (context, question, answer)."""
        audio_file_name = list_files(path)[index]
        spect = mel_spectrogram(os.path.join(path, audio_file_name), resampled=True, max_len=15, normalize=True)
        label = self.labels[os.path.split(audio_file_name)[1]]
        return spect, label, audio_file_name

    def __len__(self):
        return len(list_files(os.path.join(self.root, str(self.objid))))


def lstm_collate_fn(data):
    # Sort a data list by length (descending order).
    data.sort(key=lambda x: x[2].shape[0], reverse=True)
    spect, label, transcript_feature = zip(*data)
    lengths = [t.shape[0] for t in transcript_feature]
    lengths = torch.LongTensor(sorted(lengths)[::-1])
    padded_transcripts = torch.zeros(len(transcript_feature), max(lengths)) #, num_coeffs)
    for i, transcript in enumerate(transcript_feature):
        end = lengths[i]
        padded_transcripts[i, :end] = torch.from_numpy(transcript[:end])
    spect = torch.FloatTensor(spect)
    label = torch.FloatTensor(label)
    return spect, label, padded_transcripts, lengths