import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class BERT(nn.Module):
    """BERT

    Args:
        num_classes (int): Total number of classes.
        lm_weight (str): Pretrained model name or path. Defaults to 'bert-base-cased'.
    """
    def __init__(
        self,
        num_classes,
        lm_weight='bert-base-cased',
        **kwargs
    ):
        super().__init__()
        self.lm = AutoModelForSequenceClassification.from_pretrained(lm_weight,
                                                                     num_labels=num_classes,
                                                                     torchscript=True)

    def forward(self, input):
        input_ids = input['text'] # (batch_size, sequence_length)
        x = self.lm(input_ids, attention_mask=input_ids != self.lm.config.pad_token_id)[0] # (batch_size, num_classes)
        return {'logits': x}
