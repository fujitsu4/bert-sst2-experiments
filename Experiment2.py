# Imports
import torch
import pandas as pd
import string
from transformers import BertTokenizer, BertModel, BertConfig
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import os
import re
from collections import defaultdict

# Number of iterations (different initializations)
NUM_ITERATIONS = 10

# Seed setting function
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load the dataset once
print("\n Loading the dataset")
dataset = load_dataset("glue", "sst2")
test_sentences = dataset['validation']['sentence'][:100]

# Dictionary to store observed features by layer, level, and iteration
feature_split_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for iteration in range(NUM_ITERATIONS):
    print(f"\n Iteration {iteration + 1}/{NUM_ITERATIONS}")
    set_seed(42 + iteration)  # Different seed for each run

    # Standard tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create an untrained BERT model
    config = BertConfig(output_attentions=True)
    model = BertModel(config)
    model.eval()
    model.init_weights()

    for couche_number in range(12):
        print(f"\n Processing layer {couche_number + 1}")
        data = []

        for sent_idx, sentence in enumerate(test_sentences):
            inputs = tokenizer(sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            attention = outputs.attentions[couche_number]  # Current layer
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attention = attention[0]  # (heads=12, q_len, k_len)

            for head_idx, head_matrix in enumerate(attention):
                col_sums = head_matrix.sum(dim=0).tolist()
                top5_indices = sorted(range(len(col_sums)), key=lambda i: col_sums[i], reverse=True)[:5]

                for token_idx, token_text in enumerate(tokens):
                    is_cls = int(token_text == "[CLS]")
                    is_sep = int(token_text == "[SEP]")
                    total_attention_received = col_sums[token_idx]
                    is_top5 = int(token_idx in top5_indices)
                    is_punctuation = int(token_text in string.punctuation)
                    token_length = len(token_text)
                    is_subword = int(token_text.startswith("##"))
                    position_in_sentence = token_idx
                    is_capitalized = int(token_text[0].isupper()) if token_text and token_text[0].isalpha() else 0

                    data.append({
                        "sentence_id": sent_idx,
                        "head": head_idx,
                        "token_text": token_text,
                        "is_cls": is_cls,
                        "is_sep": is_sep,
                        "total_attention_received": total_attention_received,
                        "is_top5": is_top5,
                        "is_punctuation": is_punctuation,
                        "token_length": token_length,
                        "is_subword": is_subword,
                        "position_in_sentence": position_in_sentence,
                        "is_capitalized": is_capitalized
                    })

        df = pd.DataFrame(data)
        le_token = LabelEncoder()
        df['token_encoded'] = le_token.fit_transform(df['token_text'])

        X = df[['token_encoded', 'is_cls', 'is_sep', 'is_punctuation', 'token_length', 'is_subword', 'position_in_sentence', 'is_capitalized']]
        y = df['is_top5']

        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(X, y)

        tree_rules = export_text(clf, feature_names=list(X.columns))
        output_dir = "outputs_iterations"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"iteration_{iteration + 1}_couche_{couche_number + 1}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(tree_rules)

        # Rule analysis: extract splits by level
        lines = tree_rules.splitlines()
        profondeur_par_ligne = []
        features_splits = []
        for line in lines:
            match = re.search(r'( *)(\|---|\|   \|---)?\s*(\w+)', line)
            if match:
                espaces = len(match.group(1))
                feature = match.group(3)
                profondeur = espaces // 4
                profondeur_par_ligne.append(profondeur)
                features_splits.append((profondeur, feature))

        for profondeur, feature in features_splits:
            if profondeur < 4:
                feature_split_stats[couche_number + 1][profondeur + 1][feature] += 1

# Final summary of results
print("\n Compiling split statistics")
all_records = []
for couche, niveaux in feature_split_stats.items():
    for niveau, feature_counts in niveaux.items():
        total = sum(feature_counts.values())
        for feature, count in feature_counts.items():
            all_records.append({
                "Layer": couche,
                "Level": niveau,
                "Feature": feature,
                "Occurrences": count,
                "% Occurrences": round(100 * count / total, 2)
            })

df_stats = pd.DataFrame(all_records)
df_stats.sort_values(by=["Layer", "Level", "% Occurrences"], ascending=[True, True, False], inplace=True)
df_stats.to_csv("feature_split_statistics.csv", index=False)

print("\n Analysis completed and saved to 'feature_split_statistics.csv'")