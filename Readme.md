# Hate Speech Classification Task

## Overview

This project classifies tweets as "Hate" or "Normal" using a Bidirectional LSTM in PyTorch.

## Project Structure

- `src/`: Contains modular code for Data Loading and Model Architecture.
- `notebook.ipynb`: Walkthrough of EDA, Training, and Evaluation.
- `submission.csv`: Final predictions on test data.

## Approach

1. **Preprocessing:** Cleaned @mentions, and special characters.
2. **Data Pipeline:** Custom PyTorch Dataset with dynamic padding.
3. **Model:** Bidirectional LSTM with Embedding layer.
   - _Why:_ Captures context from both directions (e.g., sarcasm).

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook `notebook.ipynb`.
