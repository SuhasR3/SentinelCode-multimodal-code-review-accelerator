# getting the model and make it ready for the training

from src.models.bert_classifier import build_model

# waht we can use for good/bad software detection/quality classification
# 1) "microsoft/codebert-base" (Best for:Defects4J,Bugs.jar,Function-level classification)
# 2) "microsoft/codebert-base-mlm" (best if we want to pretrain before fine tunning)
# 3) "microsoft/graphcodebert-base" (Best for:Bug detection, Code vulnerability classification,When semantic structure matters) 
# 4) "huggingface/CodeBERTa-small-v1" ( Best for:Fast experiments, Limited GPU, Quick iteration)
### I guess the best option would be the 4th one + freezing the transformer since we have no GPU to fine tune. with shorter sequence length can be done
### Our choice is dictated by the computational power mostly...correct me if I am wrong

MODEL_NAME = "huggingface/CodeBERTa-small-v1"

# how many classes we have
NUM_LABELS = 2

# Dataset is small → dropout increase (0.2–0.3)
# Dataset is large → dropout default 0.1 
DROPOUT = 0.1


model = build_model(
    pretrained_name=MODEL_NAME,
    num_labels=NUM_LABELS,
    dropout=DROPOUT
)
