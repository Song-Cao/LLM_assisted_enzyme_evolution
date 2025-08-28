import csv
import random
from Bio import pairwise2
import torch
from transformers import EsmModel, EsmTokenizer
import pickle

# Constants
TRAIN_TEST_RATIO = 0.8
MAX_OCCURRENCES = 5
SEQUENCE_IDENTITY_THRESHOLD = 0.6

# Function to calculate sequence identity
def calculate_sequence_identity(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best_alignment = alignments[0]
    aligned_seq1, aligned_seq2 = best_alignment.seqA, best_alignment.seqB
    matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2))
    return matches / max(len(aligned_seq1), len(aligned_seq2))

# Function to split dataset into training and testing sets
def split_train_test(dataset_rows):
    train_set = []
    test_set = []
    reaction_count_train = {}
    reaction_count_test = {}
    enzyme_count_train = {}
    enzyme_count_test = {}

    random.shuffle(dataset_rows)

    for row in dataset_rows:
        reaction_id_A = row['reactionA_id']
        reaction_id_B = row['reactionB_id']
        enzymeA_seq = row['enzymeA_sequence']
        enzymeB_seq = row['enzymeB_sequence']

        train_ratio = len(train_set) / len(dataset_rows)

        # Determine target set based on desired train/test ratio
        if train_ratio < TRAIN_TEST_RATIO:
            target_set = train_set
            reaction_count = reaction_count_train
            enzyme_count = enzyme_count_train
        else:
            target_set = test_set
            reaction_count = reaction_count_test
            enzyme_count = enzyme_count_test

        # Check if adding the row exceeds max occurrences
        if (reaction_count.get(reaction_id_A, 0) < MAX_OCCURRENCES and
            reaction_count.get(reaction_id_B, 0) < MAX_OCCURRENCES and
            enzyme_count.get(enzymeA_seq, 0) < MAX_OCCURRENCES and
            enzyme_count.get(enzymeB_seq, 0) < MAX_OCCURRENCES):

            # Ensure sequence identity condition for test set
            if target_set == test_set:
                if all(calculate_sequence_identity(enzymeA_seq, train_enzyme) < SEQUENCE_IDENTITY_THRESHOLD
                       for train_enzyme in enzyme_count_train.keys()) and \
                   all(calculate_sequence_identity(enzymeB_seq, train_enzyme) < SEQUENCE_IDENTITY_THRESHOLD
                       for train_enzyme in enzyme_count_train.keys()):
                    target_set.append(row)
                    reaction_count[reaction_id_A] = reaction_count.get(reaction_id_A, 0) + 1
                    reaction_count[reaction_id_B] = reaction_count.get(reaction_id_B, 0) + 1
                    enzyme_count[enzymeA_seq] = enzyme_count.get(enzymeA_seq, 0) + 1
                    enzyme_count[enzymeB_seq] = enzyme_count.get(enzymeB_seq, 0) + 1
            else:
                target_set.append(row)
                reaction_count[reaction_id_A] = reaction_count.get(reaction_id_A, 0) + 1
                reaction_count[reaction_id_B] = reaction_count.get(reaction_id_B, 0) + 1
                enzyme_count[enzymeA_seq] = enzyme_count.get(enzymeA_seq, 0) + 1
                enzyme_count[enzymeB_seq] = enzyme_count.get(enzymeB_seq, 0) + 1

    return train_set, test_set

# Function to create embeddings for enzyme sequences
def create_embeddings(sequences, model, tokenizer):
    embeddings = {}
    for seq in sequences:
        inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=False)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings[seq] = outputs.last_hidden_state.squeeze().numpy()
    return embeddings

# Main function
def main():
    # Load dataset
    with open('dataset.csv', 'r') as file:
        reader = csv.DictReader(file)
        dataset_rows = list(reader)

    # Split dataset
    train_set, test_set = split_train_test(dataset_rows)

    # Write training set to CSV
    with open('train_dataset.csv', 'w', newline='') as train_file:
        fieldnames = dataset_rows[0].keys()
        writer = csv.DictWriter(train_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_set)

    # Write testing set to CSV
    with open('test_dataset.csv', 'w', newline='') as test_file:
        fieldnames = dataset_rows[0].keys()
        writer = csv.DictWriter(test_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_set)

    # Extract unique enzyme sequences
    unique_sequences = set(row['enzymeA_sequence'] for row in dataset_rows)
    unique_sequences.update(row['enzymeB_sequence'] for row in dataset_rows)
    unique_sequences.update(row['aligned_enzymeA'] for row in dataset_rows)
    unique_sequences.update(row['aligned_enzymeB'] for row in dataset_rows)

    # Load ESM-2 model and tokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Create embeddings
    embeddings = create_embeddings(unique_sequences, model, tokenizer)

    # Save embeddings to file
    with open('enzyme_embeddings.pkl', 'wb') as embed_file:
        pickle.dump(embeddings, embed_file)

if __name__ == "__main__":
    main()
