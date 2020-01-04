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

        # Note that that mutates labels. This is only okay because we don't use reuse labels in
        # the learner. If this ever changes you can fix this by using labels.clone()
        if labels is not None:
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i][j] == self.crf.num_tags:
                        # Change 'X' label to 'O' so crf doesn't try to access wrong index
                        # Using a mask does not fix this.
                        labels[i][j] = 0
            outputs = (-self.crf(emissions=outputs[1], tags=labels), outputs[1])

        return outputs
