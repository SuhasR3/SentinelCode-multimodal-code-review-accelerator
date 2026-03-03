
from transformers import AutoConfig, AutoModelForSequenceClassification

# the model will need as input 3 things:
# 1) pretrained_name, a string: name of the model (eg. "bert-base-uncased")
# 2) num_labels, integer with the number of classes to predict (probably 2 if we wnat to detect good software and bad software)
# 3) dropout, float with the regularization strength (eg. 0.1)
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