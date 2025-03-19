import numpy as np
import random


text_data = "The quick brown fox jumps over the lazy dog. Handwritten text generation is interesting!"

# Character set
chars = sorted(set(text_data))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

# Prepare input sequences
sequence_length = 5
data_sequences = []
data_labels = []

for i in range(len(text_data) - sequence_length):
    seq = text_data[i:i+sequence_length]
    label = text_data[i+sequence_length]
    data_sequences.append([char_to_index[c] for c in seq])
    data_labels.append(char_to_index[label])

# Convert to numpy arrays
X = np.array(data_sequences)
y = np.array(data_labels)

# Simple Markov-based model for text generation
def generate_text(seed_text, next_chars=50):
    generated = seed_text
    for _ in range(next_chars):
        last_chars = generated[-sequence_length:]
        if any(c not in char_to_index for c in last_chars):
            break
        idx = random.choice([char_to_index[c] for c in chars])
        generated += index_to_char[idx]
    return generated

# Interactive text generation
while True:
    seed = input("Enter seed text (or type 'exit' to quit): ")
    if seed.lower() == 'exit':
        break
    print("Generated Text:", generate_text(seed))
