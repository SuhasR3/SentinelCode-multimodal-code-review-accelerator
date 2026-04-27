# SentinelCode

SentinelCode is an AI-assisted code review triage system for Java snippets. It combines:

- a semantic classifier based on a fine-tuned transformer checkpoint
- a structural branch based on extracted software metrics
- a late-fusion decision layer
- an optional LLM explanation layer for plain-language review output

The goal is not to replace human code review. The goal is to surface likely risk faster, explain why a snippet may deserve attention, and help reviewers prioritize what to inspect first.

## What The Project Does

Given a code snippet, SentinelCode:

1. extracts structural metrics such as:
   - `loc`
   - `v(g)` cyclomatic complexity
   - `branchCount`
   - Halstead-style metrics like `v`, `d`, `e`, `uniq_Op`, `uniq_Opnd`
2. runs a semantic classification model over the raw code
3. runs a structural MLP over the extracted metrics
4. fuses both outputs into a final buggy/clean prediction
5. optionally sends the snippet plus model outputs to an LLM for a review-style explanation
6. presents a combined final review priority in the frontend

## Current Architecture

### 1. Semantic Branch

- Uses `transformers` sequence classification over code text
- Loads from `outputs/checkpoints/code-model-best`
- Produces the semantic buggy probability

Main files:

- [src/train.py](src/train.py)
- [src/predict.py](src/predict.py)
- [src/eval.py](src/eval.py)

### 2. Structural Branch

- Extracts structural and token-derived metrics directly from the snippet
- Uses a PROMISE-style MLP checkpoint
- Produces the structural buggy probability

Main files:

- [Final/src/final_user_input.py](Final/src/final_user_input.py)
- [Final/src/promise_model.py](Final/src/promise_model.py)

### 3. Late Fusion

- Combines semantic and structural probabilities using a tunable `alpha`
- `alpha` controls how much weight the semantic branch receives
- The frontend exposes this directly as the “Fusion Weight”

### 4. LLM Explanation Layer

- Optional post-prediction explanation step
- Uses the OpenAI Responses API through a local `.env` configuration
- Produces a plain-language review with sections like verdict, summary, likely issue, and recommended action

Important design choice:

- the LLM is not part of the classifier
- the model handles detection
- the LLM handles explanation

### 5. Final Review Priority

The frontend shows a decision layer that combines:

- model signal
- LLM security reasoning
- final review priority

This keeps the classifier’s confidence visible while still surfacing a practical triage outcome for demo and review use.

## Frontend

The current frontend is a Flask app in the `Final/` folder.

Main files:

- [Final/webapp.py](Final/webapp.py)
- [Final/templates/index.html](Final/templates/index.html)
- [Final/static/styles.css](Final/static/styles.css)

The UI currently includes:

- code input panel
- semantic/structural fusion weight slider
- optional LLM review toggle
- decision layer with final review priority
- prediction summary
- branch probability bars
- extracted metrics panel
- formatted LLM review cards

## Project Structure

```text
SentinelCode-multimodal-code-review-accelerator/
├── Final/
│   ├── webapp.py
│   ├── .env.example
│   ├── static/
│   │   └── styles.css
│   ├── templates/
│   │   └── index.html
│   ├── data/
│   │   └── processed/
│   ├── outputs/
│   │   └── checkpoints/
│   └── src/
│       ├── final_user_input.py
│       ├── final_test.py
│       └── promise_model.py
├── data/
│   ├── codesearchnet/
│   ├── load_codesearchnet.py
│   ├── load_dataset_Bugs_jar.py
│   ├── load_dataset_Defects4J.py
│   ├── load_promise.py
│   └── preprocess_datasets.py
├── outputs/
│   ├── checkpoints/
│   └── pretrained_codesearchnet/
├── src/
│   ├── configs/
│   ├── data/
│   ├── models/
│   ├── utils/
│   ├── eval.py
│   ├── predict.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate your virtual environment, then install dependencies:

```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

## Environment Configuration

Create a local environment file in `Final/.env`:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

Notes:

- `Final/.env.example` is only a template
- the real `.env` file should stay local and should not be committed

## Running The Frontend

Start the Flask app from the repo root:

```powershell
cd Final
..\venv\Scripts\python.exe webapp.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Running CLI Inference

From the `Final/` directory:

```powershell
..\venv\Scripts\python.exe -m src.final_user_input --text "public int divide(int a, int b) { return a / b; }"
```

With LLM explanation:

```powershell
..\venv\Scripts\python.exe -m src.final_user_input --text "public int divide(int a, int b) { return a / b; }" --explain-with-llm
```

You can also increase the output budget if the LLM review gets truncated:

```powershell
..\venv\Scripts\python.exe -m src.final_user_input --text "..." --explain-with-llm --llm-max-output-tokens 1800
```

## Training / Evaluation

The root `src/` folder contains the model training and evaluation scripts for the semantic branch.

Examples:

```powershell
.\venv\Scripts\python.exe -m src.train
.\venv\Scripts\python.exe -m src.eval
.\venv\Scripts\python.exe -m src.predict --text "public int add(int a, int b) { return a + b; }"
```

Note:

- the repo currently contains model checkpoints and utility scripts, but local dataset state may differ by machine
- some historical README references from older architecture drafts are no longer accurate

## Example Demo Flow

1. Paste a code snippet into the frontend
2. Adjust the semantic fusion weight if needed
3. Run analysis
4. Inspect:
   - final review priority
   - prediction summary
   - branch probabilities
   - extracted metrics
   - LLM review

Good demo contrast:

- clean code example for low priority
- divide-by-zero or null dereference example for moderate concern
- SQL injection example for high review priority

## Current Limitations

- classifier confidence is currently threshold-based and simple
- the structural branch relies on extracted metrics rather than full program context
- LLM review quality depends on the configured model and prompt-following behavior
- short snippets can produce low-confidence model outputs even when human reasoning is strong
- the available local dataset files may not fully match the training split used to produce the current checkpoints

## Recommended Positioning

SentinelCode should be presented as:

- an AI first-pass reviewer
- a code review triage assistant
- a human-in-the-loop review accelerator

It should not be presented as:

- a perfect bug detector
- a replacement for human review
- a guarantee that code is secure or defect-free
