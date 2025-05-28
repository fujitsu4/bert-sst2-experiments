# Imports
import torch
import pandas as pd
import re
import string
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

layer_number = 0 # Layer 1

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)
model.eval()

from datasets import load_dataset
# Temporarily disable caching
#set_caching_enabled(False)
dataset = load_dataset("glue", "sst2")

test_sentences = dataset['validation']['sentence'][:100]  # First 100 sentences

# Function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Apply to all sentences
test_sentences_no_punct = [remove_punctuation(sent) for sent in test_sentences]

# Display the first 100 sentences without punctuation
print("\n🔹 First sentences without punctuation:\n")
for i, sent in enumerate(test_sentences_no_punct):
    print(f"{i+1:3d}: {sent}")

# Extract attention and compute column-wise sums
data = []

for sent_idx, sentence in enumerate(test_sentences_no_punct):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    attention = outputs.attentions[layer_number]  # Specified layer
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # attention: (batch=1, heads=12, q_len, k_len)
    attention = attention[0]  # (heads=12, q_len, k_len)

    for head_idx, head_matrix in enumerate(attention):  # head_matrix: (q_len, k_len)
        # Column-wise sum (total attention received by each token)
        col_sums = head_matrix.sum(dim=0)  # sum over all queries for each key
        col_sums = col_sums.tolist()

        # Identify top 5 most attended-to tokens
        top5_indices = sorted(range(len(col_sums)), key=lambda i: col_sums[i], reverse=True)[:5]

        for token_idx, token_text in enumerate(tokens):
            is_cls = int(token_text == "[CLS]")
            is_sep = int(token_text == "[SEP]")
            total_attention_received = col_sums[token_idx]
            is_top5 = int(token_idx in top5_indices)

            # Additional attributes
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

# Save as DataFrame + CSV
df = pd.DataFrame(data)
print("\n Preview of the new dataset:")
print(df.head(10))

df.to_csv("attention_sum_dataset.csv", index=False)
print("\n CSV file saved as 'attention_sum_dataset.csv'")

# Train a Decision Tree to predict is_top5
# Load CSV (optional if everything is in the same script)
df = pd.read_csv("attention_sum_dataset.csv")

# Encode token_text for use as a feature
le_token = LabelEncoder()
df['token_encoded'] = le_token.fit_transform(df['token_text'])

# Display token mapping
token_mapping = dict(zip(le_token.classes_, le_token.transform(le_token.classes_)))
print("\nToken to encoded mapping:")
for token_text, token_encoded in token_mapping.items():
    print(f"{token_text} --> {token_encoded}")

# Selected features
X = df[['token_encoded', 'is_cls', 'is_sep', 'is_punctuation', 'token_length', 'is_subword', 'position_in_sentence', 'is_capitalized']]
y = df['is_top5']

# Train the decision tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

print("\n Layer number:\n", layer_number + 1)
# Display tree rules
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\n Decision Tree rules:\n")
print(tree_rules)

# Save tree rules to .txt file
output_filename = f"output_layer_{layer_number + 1}_no_punctuation.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(tree_rules)

print(f"\n Decision Tree rules saved to '{output_filename}'")

# Initialize list to store summary results
results = []

# Read rule file for each layer
for layer_number in range(1, 13):  # layers 1 to 12
    filename = f"output_layer_{layer_number}_no_punctuation.txt"

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Extract level 1–4 splits
        splits = []
        depths = []

        for line in lines:
            match = re.search(r'( *)(\|---|\|   \|---)?\s*(\w+)', line)
            if match:
                spaces = len(match.group(1))
                feature = match.group(3)

                depth = spaces // 4
                depths.append(depth)
                splits.append(feature)

        # Store level 1 to 4 splits
        split_level_1 = splits[0] if len(splits) > 0 else None
        split_level_2 = splits[1] if len(splits) > 1 else None
        split_level_3 = splits[2] if len(splits) > 2 else None
        split_level_4 = splits[3] if len(splits) > 3 else None

        max_depth = max(depths) if depths else 0

        results.append({
            "Layer": layer_number,
            "Level 1": split_level_1,
            "Level 2": split_level_2,
            "Level 3": split_level_3,
            "Level 4": split_level_4,
            "Max Depth": max_depth
        })

    except FileNotFoundError:
        print(f" File {filename} not found, skipping.")

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results to CSV
df_results.to_csv("split_comparison_and_depth_per_layer.csv", index=False)

print("\n Analysis completed and saved to 'split_comparison_and_depth_per_layer.csv'!")
