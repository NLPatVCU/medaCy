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

        if labels is None:
            return outputs

        # If labels were given, we'll want to use them to train the CRF layer as well
        labels = labels.clone()
        labels[labels == self.crf.num_tags] = 0

        # After 'X' labels have been removed, pass the emission scores through the CRF layer
        # Returns CRF output if there were labels or original emission scores otherwise
        return -self.crf(emissions=outputs[1], tags=labels), outputs[1]
