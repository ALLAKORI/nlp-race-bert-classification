# Educational Text Classification with Fine-tuned BERT (RACE)

NLP course project (ENSA Beni Mellal, cycle FI-IACS, January 2026), team of 4:
ADJEVI Kodjo Marius-Ben, ELJARIDA Saad Eddine, **ALLADO Kossi Richard**, KPADENOU Kossi Roland.

## Goal

Answer multiple-choice reading comprehension questions automatically: given an educational
passage and a question, classify the correct option among A/B/C/D using a Transformer model
fine-tuned on the [RACE dataset](https://arxiv.org/abs/1704.04683) (Reading Comprehension
from Examinations).

## Approach

- Model: **DistilBERT-base-uncased** (lightweight BERT, 67M parameters) chosen to fit
  Google Colab GPU constraints.
- Multiple-choice adaptation: HuggingFace `AutoModelForMultipleChoice`; for each question,
  4 input sequences are encoded as `[CLS] Article [SEP] Question + Option_i [SEP]` and scored
  in parallel. Softmax over the 4 options selects the answer.
- Preprocessing: text cleaning (spaces, punctuation, quote normalization), WordPiece
  tokenization, max length 384 tokens with `only_first` truncation on the article.
- Training: PyTorch + HuggingFace Transformers on Google Colab. AdamW (lr 2e-5), batch 16,
  3 epochs, 10% warmup steps, weight decay 0.01, gradient clipping 1.0, cross-entropy loss,
  early stopping on best validation accuracy.
- Data subset due to compute limits: 5,000 train / 1,000 validation / 1,000 test examples.

## Evaluation

| Metric | Score (1k test subset) |
|---|---|
| Accuracy | **36.10%** |
| F1-score (macro) | **35.53%** |

Above the random baseline (25% on 4 options), below the state of the art because the model
was trained on a reduced subset with limited epochs. The report documents learning curves,
the confusion matrix, and a critical error analysis: lexical confusions, multi-step inference
questions handled poorly, and information lost when passages exceed the 512-token limit.

## Repository structure

- `report/` — detailed project report in French (dataset statistics, methodology, results,
  limitations, perspectives).
- `tp1-text-processing/` — individual TP on classic text preprocessing: sentence
  segmentation, tokenization with regex, batch processing of `.txt` corpora to segmented
  JSON, corpus statistics tables (paragraphs / sentences / tokens / characters).

Note: the fine-tuning notebook was executed directly on Google Colab. The archived
deliverable is the detailed report; a notebook export may be added later if recovered.

## Skills illustrated

NLP, Transformers fine-tuning, HuggingFace, PyTorch, multiple-choice classification,
text preprocessing, WordPiece tokenization, evaluation methodology (accuracy, macro F1,
confusion matrix), error analysis and critical reporting.
