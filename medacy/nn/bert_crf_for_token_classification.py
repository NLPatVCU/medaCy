from transformers import BertForTokenClassification
import torch
from torchcrf import CRF

class BertCrfForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels
        )

        if labels is not None:
            mask = []
            for i in range(labels.shape[0]):
                sequence_mask = []
                for j in range(labels.shape[1]):
                    if labels[i][j] == self.crf.num_tags:
                        sequence_mask.append(0)
                        labels[i][j] = 0
                    else:
                        sequence_mask.append(1)
                mask.append(sequence_mask)
            mask = torch.tensor(mask, dtype=torch.uint8)
            outputs = (self.crf(emissions=outputs[1], tags=labels, mask=mask), outputs[1])

        return outputs
