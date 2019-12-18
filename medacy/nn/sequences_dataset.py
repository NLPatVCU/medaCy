import torch
from torch.utils.data import Dataset

class SequencesDataset(Dataset):
    def __init__(self, device, sequences, sequence_labels=None):
        self.device = device
        self.sequences = sequences
        self.sequence_labels = sequence_labels

    def __len__(self):
        return len(self.sequences)

    def collate(self, samples):
        max_length = 0
        collated_samples = [[], []]

        if self.sequence_labels is not None:
            max_length = max([len(sample[0]) for sample in samples])
            collated_samples.append([])
        else:
            max_length = max([len(sample) for sample in samples])

        for sample in samples:
            sequence, labels = [], []

            if self.sequence_labels is not None:
                sequence, labels = sample
                labels = labels.tolist()
            else:
                sequence = sample

            sequence = sequence.copy()
            padding_length = max_length - len(sequence)

            if self.sequence_labels is None:
                attention_mask = [1] * len(sequence)
            else:
                attention_mask = [int(x != labels[0]) for x in labels]

            padding = [0] * padding_length
            sequence.extend(padding)
            attention_mask.extend(padding)

            collated_samples[0].append(sequence)
            collated_samples[1].append(attention_mask)

            if self.sequence_labels is not None:
                labels.extend(padding)
                collated_samples[2].append(labels)

        return [torch.tensor(x, device=self.device) for x in collated_samples]

    def __getitem__(self, idx):
        if self.sequence_labels is None:
            return self.sequences[idx]
        else:
            return (self.sequences[idx], self.sequence_labels[idx])
