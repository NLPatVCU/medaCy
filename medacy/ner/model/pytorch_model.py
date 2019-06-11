import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
from spacy.gold import biluo_tags_from_offsets
from medacy.data import Dataset
from medacy.tools import Annotations

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class PytorchModel:
    def __init__(self, bidirectional=False):
        self.bidirectional = bidirectional

        torch.manual_seed(1)

    def prepare_sequence(self, sequence, to_index):
        indexes = [to_index[w] for w in sequence]
        return torch.tensor(indexes, dtype=torch.long)

    def devectorize(self, vectors, to_index):
        to_tag = {y:x for x, y in to_index.items()}
        tags = []

        for vector in vectors:
            max_value = max(vector)
            index = list(vector).index(max_value)
            tags.append(to_tag[index])
            
        return tags

    def fit(self, dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError("Must pass in an instance of Dataset containing your training files")

        labels = dataset.get_labels()
        training_data = dataset.get_training_data('pytorch')

        word_to_ix = {}
        for sent, tags in training_data:
            for word in sent:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        # print('WORD TO IX')
        # print(word_to_ix)

        # tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
        tag_to_ix = {"O": 0}

        for label in labels:
            tag_to_ix[label] = len(tag_to_ix)

        # print('TAG TO IX')
        # print(tag_to_ix)

        # These will usually be more like 32 or 64 dimensional.
        # We will keep them small, so we can see how the weights change as we train.
        EMBEDDING_DIM = 64
        HIDDEN_DIM = 64

        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), self.bidirectional)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # See what the scores are before training
        # Note that element i,j of the output is the score for tag j for word i.
        # Here we don't need to train, so the code is wrapped in torch.no_grad()
        with torch.no_grad():
            inputs = self.prepare_sequence(training_data[0][0], word_to_ix)
            tag_scores = model(inputs)
            print('TAG SCORES')
            print(tag_scores)

        for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
            print('Epoch %d' % epoch)
            losses = []
            for sentence, tags in training_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                optimizer.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = self.prepare_sequence(sentence, word_to_ix)
                targets = self.prepare_sequence(tags, tag_to_ix)

                # Step 3. Run our forward pass.
                tag_scores = model(sentence_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer.step()
                # print(loss)
                losses.append(loss)

            average_loss = sum(losses) / len(losses)
            print('AVG LOSS: %f' % average_loss)

        # See what the scores are after training
        with torch.no_grad():
            inputs = self.prepare_sequence(training_data[0][0], word_to_ix)
            print(inputs)
            tag_scores = model(inputs)

            # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
            # for word i. The predicted tag is the maximum scoring tag.
            # Here, we can see the predicted sequence below is 0 1 2 0 1
            # since 0 is index of the maximum value of row 1,
            # 1 is the index of maximum value of row 2, etc.
            # Which is DET NOUN VERB DET NOUN, the correct sequence!
            print('SECOND TAG SCORES')
            print(tag_scores)
            print('----TRAINING------')
            print(training_data[0][1])
            print('----PREDICTION----')
            print(self.devectorize(tag_scores, tag_to_ix))
