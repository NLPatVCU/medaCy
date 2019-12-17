import torch
from torch.utils.data import Dataset

class SequencesDataset(Dataset):
    def __init__(self, device, sequences, sequence_labels=None):
        self.device = device
        self.sequences = sequences
        self.sequence_lables = sequence_labels

    def __len__(self):
        return len(self.sequences)

    def collate(self, samples):
        collated_sequences = []
        collated_labels = []
        collated_masks = []
        max_length = max([len(sample[0]) for sample in samples])

        for sample in samples:
            sequence, labels = sample
            sequence = sequence.copy()
            labels = labels.tolist()
            padding_length = max_length - len(sequence)
            attention_mask = [1] * len(sequence)

            padding = [0] * padding_length
            attention_mask.extend(padding)
            sequence.extend(padding)
            labels.extend(padding)

            collated_sequences.append(sequence)
            collated_labels.append(labels)
            collated_masks.append(attention_mask)

        collated_samples = (
            torch.tensor(collated_sequences, device=self.device),
            torch.tensor(collated_labels, device=self.device),
            torch.tensor(collated_masks, device=self.device)
        )

        return collated_samples

    def __getitem__(self, idx):
        if self.sequence_lables is None:
            return self.sequences[idx]
        else:
            return (self.sequences[idx], self.sequence_lables[idx])
