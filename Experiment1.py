# Imports
import torch
import pandas as pd
import string
import pandas as pd
import re
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

layer_number = 0  # Layer 1

# 1. Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)
model.eval()

# 2. Load a data sample
dataset = load_dataset("glue", "sst2")
test_sentences = dataset['validation']['sentence'][:100]  # First 100 sentences from SST-2 dataset

# 3. Extract attention and compute column sums
data = []

for sent_idx, sentence in enumerate(test_sentences):
    # print(f"\n--- Sequence {sent_idx + 1} : {sentence} ---")
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    attention = outputs.attentions[layer_number]  # Layer 1 (index 0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # attention: (batch=1, heads=12, q_len, k_len)
    attention = attention[0]  # (heads=12, q_len, k_len)

    for head_idx, head_matrix in enumerate(attention):  # head_matrix: (q_len, k_len)
        # Column sums (total attention received by each token)
        col_sums = head_matrix.sum(dim=0)  # sum over all queries for each key
        col_sums = col_sums.tolist()

        # Find indices of top 5 most attended tokens
        top5_indices = sorted(range(len(col_sums)), key=lambda i: col_sums[i], reverse=True)[:5]

        for token_idx, token_text in enumerate(tokens):
            is_cls = int(token_text == "[CLS]")
            is_sep = int(token_text == "[SEP]")
            total_attention_received = col_sums[token_idx]
            is_top5 = int(token_idx in top5_indices)

            # New features requested
            is_punctuation = int(token_text in string.punctuation)
            token_length = len(token_text)
            is_subword = int(token_text.startswith("##"))
            position_in_sentence = token_idx  # token's position in the sentence
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

# 4. Save as DataFrame + CSV
df = pd.DataFrame(data)
print("\n✅ Preview of the new dataset:")
print(df.head(10))

df.to_csv("attention_sum_dataset.csv", index=False)
print("\n✅ CSV file saved as 'attention_sum_dataset.csv'")

# 5. Train a Decision Tree to predict is_top5
# Load CSV (optional if running everything in one script)
df = pd.read_csv("attention_sum_dataset.csv")

# Encode token_text as a feature
le_token = LabelEncoder()
df['token_encoded'] = le_token.fit_transform(df['token_text'])

# Display mapping table
token_mapping = dict(zip(le_token.classes_, le_token.transform(le_token.classes_)))
print("\nMapping table token_text -> token_encoded:")
for token_text, token_encoded in token_mapping.items():
    print(f"{token_text} --> {token_encoded}")

# 6. Updated features
X = df[['token_encoded', 'is_cls', 'is_sep', 'is_punctuation', 'token_length', 'is_subword', 'position_in_sentence', 'is_capitalized']]
y = df['is_top5']

# Train the tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

print("\n Layer number:\n", layer_number + 1)
# Display tree rules
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\n Decision Tree Rules:\n")
print(tree_rules)

# Save rules to a .txt file
output_filename = f"output_layer_{layer_number + 1}.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(tree_rules)

print(f"\n Rules saved in '{output_filename}'")

# Initialize a list to store results
results = []

# Read each layer file
for layer_number in range(1, 13):  # layers 1 to 12
    filename = f"output_layer_{layer_number}_untrained.txt"

    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Extract splits at levels 1 to 4
        splits = []
        depths = []  # New list to store depths

        for line in lines:
            # Look for lines containing a split
            match = re.search(r'( *)(\|---|\|   \|---)?\s*(\w+)', line)
            if match:
                spaces = len(match.group(1))  # number of spaces
                feature = match.group(3)

                # Compute depth based on indentation
                depth = spaces // 4
                depths.append(depth)
                splits.append(feature)

        # Store splits for levels 1 to 4
        split_level_1 = splits[0] if len(splits) > 0 else None
        split_level_2 = splits[1] if len(splits) > 1 else None
        split_level_3 = splits[2] if len(splits) > 2 else None
        split_level_4 = splits[3] if len(splits) > 3 else None

        # Estimate tree max depth
        max_depth = max(depths) if depths else 0

        results.append({
            "Layer": layer_number,
            "Level 1": split_level_1,
            "Level 2": split_level_2,
            "Level 3": split_level_3,
            "Level 4": split_level_4,
            "Max depth": max_depth
        })

    except FileNotFoundError:
        print(f"File {filename} not found, skipping.")

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
df_results.to_csv("comparison_splits_and_depth_by_layer.csv", index=False)

print("\n✅ Analysis completed and saved to 'comparison_splits_and_depth_by_layer.csv'!")
