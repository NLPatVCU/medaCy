import torch
from torch.utils.data import Dataset

class SequencesDataset(Dataset):
    def __init__(self, device, sequences, mask_label, sequence_labels=None, o_label=1, mappings=None):
        self.device = device
        self.sequences = sequences
        self.mask_label = mask_label
        self.sequence_labels = sequence_labels
        self.o_label = o_label
        self.mappings = mappings

    def __len__(self):
        return len(self.sequences)

    def collate(self, samples):
        max_length = 0
        collated_samples = [[], [], []]
        max_length = max([len(sample[0]) for sample in samples])

        for i, sample in enumerate(samples):
            sequence, labels = sample
            labels = labels.tolist()

            sequence = sequence.copy()
            padding_length = max_length - len(sequence)

            attention_mask = [int(x != self.mask_label) for x in labels]
            attention_mask[0] = 1
            attention_mask[-1] = 1

            sequence.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend([self.o_label] * padding_length)

            collated_samples[0].append(sequence)
            collated_samples[1].append(attention_mask)
            collated_samples[2].append(labels)

        return [torch.tensor(x, device=self.device) for x in collated_samples]

    def __getitem__(self, idx):
        if self.sequence_labels is None:
            return self.sequences[idx]
        else:
            return (self.sequences[idx], self.sequence_labels[idx])
