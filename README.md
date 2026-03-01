# SentinelCode: AI Multimodal Framework for Code Review Acceleration

SentinelCode is an AI-driven multimodal framework engineered to address the Code Review Bottleneck. While CI/CD handles automation, human code reviews remain a massive bottleneck, often forcing senior developers to waste hours auditing low-risk tweaks. This system acts as a "first responder," using a Late Fusion Multimodal Architecture to flag high-risk changes and clear low-risk code automatically.

---

## Architecture: Late Fusion Approach

A critical technical differentiator of SentinelCode is the Late Fusion approach. Rather than merging raw data at the input layer, the system processes semantic and structural streams independently. These streams are only fused at the final classification layer to ensure high-volume structural metrics do not "drown out" subtle semantic signals.

### 1. The Semantic Pipeline (Deep Learning)
* **Model**: Pretrained CodeBERT model.
* **Input**: Raw source code treated as natural language.
* **Training Data**: Datasets including Defects4J and Bugs.jar.
* **Capabilities**: Flags complex logical flaws, such as improper variable usage or insecure data flows, that traditional static analysis tools miss.

### 2. The Structural Pipeline (Numeric Metrics)
* **Method**: Extracts PROMISE-style software metrics to provide a quantitative "health check".
* **Key Metrics**:
    * **Cyclomatic Complexity & Halstead Volume**: Assesses logic density.
    * **Code Churn**: Evaluates the historical frequency of failure in specific modules.
* **Value**: Ensures the system accounts for structural fragility not apparent in raw logic alone.

---

## Prediction Targets & Benchmarks
The framework addresses two primary targets:
* **Classification**: Identifying code as "defect-prone" (Class 1) vs. "clean" (Class 0).
* **Regression**: Predicting specific severity scores or anticipated defect counts.
* **Success Metrics**: Measured by Top-K Recall and measurable reduction in Review Turnaround Time.

# Repo Structure
```text
sentinelcode-multimodal-code-review-accelerator/
├── data/                   # Raw & processed datasets (Defects4J, Bugs.jar)
├── src/
│   ├── models/
│	│	├── bert_classifier.py     	### file that fetches the BERT model
│   │   ├── semantic.py     		# CodeBERT-based NLP pipeline
│   │   ├── structural.py   		# PROMISE metric extraction & processing
│   │   └── fusion.py       		# Late Fusion classification layer
│   ├── features/
│   │   ├── extractor.py    		# Cyclomatic Complexity & Halstead Volume logic
│   │   └── churn.py        		# Git history/module failure frequency analysis
│   ├── configs/
│   │   ├── baseline.py    			### default setup/config  
│   │   └── ablation.py     		### ablation changes
│   ├── utils/
│   │   ├── explainability.py 		# SHAP & Attention visualizations
│   │   └── uncertainty.py    		# MC Dropout implementation
│	├── eval.py  					### model evaluation
│	├── predict.py  				### generate predictions on new data
│	└── train.py  					### model trainig
├── requirements.txt
└── main.py                 		# Entry point for training/inference
```