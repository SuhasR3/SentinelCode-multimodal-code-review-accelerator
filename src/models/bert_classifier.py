
from transformers import AutoConfig, AutoModelForSequenceClassification


def build_model(pretrained_name: str, num_labels: int = 2, dropout: float = 0.1):
    cfg = AutoConfig.from_pretrained(pretrained_name, num_labels=num_labels)
    # this makes our model more robust, it controls the dropout in the hidden layers / outputs, regularizes dropouts..we need to read more about this
    if hasattr(cfg, "hidden_dropout_prob"):
        cfg.hidden_dropout_prob = dropout
        
    # this is similar to the previous one just that it regulates the attention mechanism
    if hasattr(cfg, "attention_probs_dropout_prob"):
        cfg.attention_probs_dropout_prob = dropout
    #this will load the weights for the model from the input and will build a linear layer on top ....we need to check the config file
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, config=cfg)
    return model