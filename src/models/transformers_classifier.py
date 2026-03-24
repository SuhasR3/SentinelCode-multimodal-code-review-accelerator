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

    # some transformer configs also expose classifier-specific dropout
    if hasattr(cfg, "classifier_dropout"):
        cfg.classifier_dropout = dropout

    #this will load the weights for the model from the input and will build a linear layer on top ....we need to check the config file
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, config=cfg)
    return model


# added helper: freeze the transformer encoder and keep only the classification head trainable
def freeze_backbone(model) -> None:
    for param in model.parameters():
        param.requires_grad = False

    # common names for classification heads in Hugging Face models
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    if hasattr(model, "score"):
        for param in model.score.parameters():
            param.requires_grad = True


# added helper: unfreeze the last n encoder layers if we want partial fine-tuning later
def unfreeze_last_n_layers(model, n: int = 1) -> None:
    # keep the classification head trainable
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    if hasattr(model, "score"):
        for param in model.score.parameters():
            param.requires_grad = True

    # RoBERTa / CodeBERT / CodeBERTa family
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        layers = model.roberta.encoder.layer
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    # BERT family
    elif hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        layers = model.bert.encoder.layer
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True


# added helper: useful to verify how much of the model is really trainable
def count_trainable_parameters(model) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params