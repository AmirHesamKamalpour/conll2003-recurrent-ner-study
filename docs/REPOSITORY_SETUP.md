# Suggested GitHub Repository Setup

**Repository name:** `conll2003-recurrent-ner-study`

**Description:** A controlled PyTorch comparison of bidirectional RNN, LSTM, and GRU encoders for CoNLL-2003 named entity recognition.

**Topics:** `named-entity-recognition, sequence-labeling, conll2003, pytorch, bidirectional-rnn, lstm, gru, natural-language-processing`

## Before publishing

- Rename the extracted project folder to `conll2003-recurrent-ner-study` if needed.
- Upload all source files, YAML config, provided plots/reports, and `docs/assets/model_comparison.png` together with `README.md`.
- Choose a repository license yourself after checking project and dataset rights; one was not added automatically.
- If you retain `models/saved_models/*.pt`, note that the 3 weight files total roughly 19 MB; they are usable only with the matching vocabulary.
- Data download requires network access and may depend on the availability of the Parquet dataset revision.
- Do not advertise reported token macro F1 as CoNLL entity-level F1.
- The README's example clone URL contains `<YOUR_USERNAME>`; substitute your actual GitHub username after creating the repository.

## Verified facts

- Three checkpoint state dicts were loaded and confirmed to have matching dimensions: 23,627-token vocabulary, 64-dimensional embeddings, 128-dimensional output state, 9 labels.
- Python modules compile, and synthetic variable-length inputs complete a forward pass for all three encoder types.
- Saved output reports were parsed to calculate the comparative tables.
- Full dataset re-download and a fresh training reproduction were **not** performed.
