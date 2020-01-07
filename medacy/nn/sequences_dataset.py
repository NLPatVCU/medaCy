"""
PyTorch Dataset for medaCy sequences.
"""
import torch
from torch.utils.data import Dataset

class SequencesDataset(Dataset):
    """Dataset to make preprocessing and batching easier.

    :ivar device: PyTorch device to use.
    :ivar sequences: Sequences for the dataset.
    :ivar sequence_labels: True labels for sequences.
    :ivar o_label: Encoding for O label.
    :ivar x_label: Encoding for X label.
    """

    def __init__(self, device, sequences, sequence_labels, o_label, x_label):
        """Initialize dataset.

        :param device: PyTorch device to use.
        :param sequences: Sequences for the dataset.
        :param sequence_labels: True labels for sequences.
        :param o_label: Encoding for O label.
        :param x_label: Encoding for X label.
        """
        self.device = device
        self.sequences = sequences
        self.sequence_labels = sequence_labels
        self.o_label = o_label
        self.x_label = x_label

    def __len__(self):
        """Number of sequences"""
        return len(self.sequences)

    def collate(self, samples):
        """Return batch of sequences with each padded to be the length of the longest sequence in
        the batch, the proper attention mask, and the true labels.

        :param samples: Batch of sequences.
        :return: Sequences, attention masks, and labels.
        """
        # Find length of longest sequence
        max_length = max([len(sample[0]) for sample in samples])

        collated_samples = {
            'sequences': [],
            'masks': [],
            'labels': []
        }

        for sample in samples:
            sequence, labels = sample
            labels = labels.tolist() # Convert from Tensor to list

            sequence = sequence.copy() # Clone sequence to avoid mutation
            padding_length = max_length - len(sequence)

            # Create mask where item is 1 if label is not X and 0 if it is
            # Always make first and last item 1 because they should be the special tokens
            attention_mask = [int(label != self.x_label) for label in labels]
            attention_mask[0] = 1
            attention_mask[-1] = 1

            # Pad sequence, mask, and labels
            sequence.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend([self.o_label] * padding_length)

            collated_samples['sequences'].append(sequence)
            collated_samples['masks'].append(attention_mask)
            collated_samples['labels'].append(labels)

        # Convert final three lists into tensors before returning
        return [torch.tensor(x, device=self.device) for x in collated_samples.values()]

    def __getitem__(self, idx):
        """Return sequence with labels"""
        return (self.sequences[idx], self.sequence_labels[idx])
