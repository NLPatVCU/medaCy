"""
BERT model for token classification with a CRF layer.
"""
from transformers import BertForTokenClassification
from torchcrf import CRF

class BertCrfForTokenClassification(BertForTokenClassification):
    """Subclass of Transformers package BERT token classifier.
    
    :ivar crf: CRF layer.
    """
    def __init__(self, config):
        """Initialize token classification model with a CRF layer.

        :param config: Transformers config object.
        """
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
        """Pass batch through model. Change labels to be compatible with CRF. See Transformers
        documentation for parameter info:
        https://huggingface.co/transformers/v2.3.0/model_doc/bert.html#bertfortokenclassification
        """
        # Pass parameters through parent forward method to get initial outputs.
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels
        )

        # If labels were given, we'll want to use them to train the CRF layer as well
        if labels is not None:
            # Note that this mutates labels. This is only okay because we don't use reuse labels in
            # the learner. If this ever changes you can fix this by using labels.clone()
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i][j] == self.crf.num_tags:
                        # Change 'X' label to 'O' so crf doesn't try to access wrong index
                        # Using a mask does not fix this.
                        labels[i][j] = 0

            # After 'X' labels have been removed, pass the emission scores through the CRF layer
            outputs = (-self.crf(emissions=outputs[1], tags=labels), outputs[1])

        # Returns CRF output if there were labels or original emission scores otherwise
        return outputs
