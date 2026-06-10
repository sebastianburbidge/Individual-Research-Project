import os
import warnings

# 1. Silence the environment warnings cleanly
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hook into a progress bar package (built-in fallback if not installed)
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# ==========================================
# STEP 1: LOAD AND STRATIFY SAMPLE DATA
# ==========================================
print("Loading dataset and drawing 2,000-row balanced sample...")
df_raw = pd.read_csv("raw_dataset.csv")
#df_raw = df_raw[df_raw["domain"] == "academic"]
# Ensure you pull exactly 1,000 human (0) and 1,000 AI (1) texts
df_human = df_raw[df_raw["label"] == 0].sample(n=200, random_state=42)
df_ai = df_raw[df_raw["label"] == 1].sample(n=200, random_state=42)

# Combine and shuffle
df_sample = (
    pd.concat([df_human, df_ai])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

# ==========================================
# STEP 2: INITIALIZE MINDS (Models)
# ==========================================
print("Initializing NLP models (spaCy and GPT-2)...")
nlp = spacy.load("en_core_web_sm")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # Put model in evaluation mode to turn off dropout

# ==========================================
# STEP 3: FEATURE EXTRACTION FUNCTIONS
# ==========================================


def calculate_metrics(text):
    # Guard against completely blank entries
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0, 1.0, 0.0

    # A. BURSTINESS & TYPE-TOKEN RATIO (via spaCy)
    doc = nlp(text)

    # 1. TTR: Ratio of unique lemmas to total words
    lemmas = [token.lemma_.lower() for token in doc if not token.is_punct]
    ttr = len(set(lemmas)) / len(lemmas) if len(lemmas) > 0 else 0.0

    # 2. Burstiness: (Sigma - Mu) / (Sigma + Mu)
    sent_lengths = [len(sent) for sent in doc.sents]

    if len(sent_lengths) > 1:
        mu = np.mean(sent_lengths)
        sigma = np.std(sent_lengths)

        # Guard against dividing by zero if all sentences are identical length
        if (sigma + mu) > 0:
            burstiness = (sigma - mu) / (sigma + mu)
        else:
            burstiness = -1.0
    else:
        # If there's only 1 sentence, variation is technically non-existent
        burstiness = -1.0

    # B. PERPLEXITY (via GPT-2)
    # Tokenize the input string for the causal LM
    inputs = tokenizer(text, return_tensors="pt", truncate=True, max_length=1024)
    input_ids = inputs["input_ids"]

    # Wrap in no_grad to save huge chunks of memory/RAM
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.item()
        perplexity = np.exp(loss) if loss < 20 else 100000.0  # Handle infinity caps safely

    return burstiness, perplexity, ttr


# ==========================================
# STEP 4: EXECUTE THE PIPELINE LOOP
# ==========================================
print("\nProcessing text rows... (This will take a few minutes)")

burstiness_scores = []
perplexity_scores = []
ttr_scores = []

# Loop over your 2,000 texts with a visual progress indicator
for text_content in tqdm(df_sample["text"], desc="Extracting Features"):
    b, p, t = calculate_metrics(text_content)
    burstiness_scores.append(b)
    perplexity_scores.append(p)
    ttr_scores.append(t)

# Map the newly engineered features back to the structural framework
df_sample["burstiness"] = burstiness_scores
df_sample["perplexity"] = perplexity_scores
df_sample["ttr"] = ttr_scores

# ==========================================
# STEP 5: CLEAN AND EXPORT FOR R
# ==========================================
# We strip out the heavy text columns to keep the file ultra-lightweight
export_cols = ["id", "label", "burstiness", "perplexity", "ttr"]
# Fallback if 'id' column has a different name in your csv (like 'index')
if "id" not in df_sample.columns:
    df_sample["id"] = df_sample.index

df_final = df_sample[export_cols]
df_final.to_csv("ready_for_clustering.csv", index=False)

print("\nSuccess! 'ready_for_clustering.csv' has been generated.")
print("You can now boot up R Studio and read this clean 2,000-row file.")