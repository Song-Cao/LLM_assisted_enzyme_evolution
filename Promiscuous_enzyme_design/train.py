import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import pandas as pd
import pickle
from model import CATEE, train_catee
from torch_geometric.utils import from_smiles

# Function to convert SMILES to molecular graph
def get_molecular_graph(smiles):
    try:
        data = from_smiles(smiles)
        return data
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

# Function to convert fingerprint bitstring to tensor
def get_fingerprint(fp_bitstring):
    fp_array = [int(bit) for bit in fp_bitstring]
    return torch.tensor(fp_array, dtype=torch.float32)

# Function to load ESM embedding from dictionary
def get_esm_embedding(sequence, embedding_dict):
    embedding = embedding_dict.get(sequence)
    if embedding is None:
        raise ValueError(f"ESM embedding not found for sequence: {sequence}")
    return torch.tensor(embedding, dtype=torch.float32)

# Load datasets
train_df = pd.read_csv('train_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

# Load ESM embeddings
with open('enzyme_embeddings.pkl', 'rb') as f:
    esm_embeddings = pickle.load(f)

# Prepare training data
train_data = []
for _, row in train_df.iterrows():
    graph_A = get_molecular_graph(row['reactant_smiles_A'])
    graph_B = get_molecular_graph(row['reactant_smiles_B'])
    if graph_A is None or graph_B is None:
        continue
    fp_A = get_fingerprint(row['reaction_fingerprint_A'])
    fp_B = get_fingerprint(row['reaction_fingerprint_B'])
    esm_emb_A = get_esm_embedding(row['enzymeA_sequence'], esm_embeddings)
    esm_emb_align_A = get_esm_embedding(row['aligned_enzymeA'], esm_embeddings)
    aligned_seq_A = torch.tensor([ord(aa) for aa in row['aligned_enzymeA']], dtype=torch.long)
    aligned_seq_B = torch.tensor([ord(aa) for aa in row['aligned_enzymeB']], dtype=torch.long)
    train_data.append((graph_A, graph_B, fp_A, fp_B, esm_emb_A, esm_emb_align_A, aligned_seq_A, aligned_seq_B))
    # Add reversed data
    train_data.append((graph_B, graph_A, fp_B, fp_A, esm_emb_A, esm_emb_align_A, aligned_seq_B, aligned_seq_A))

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Prepare validation data
val_data = []
for _, row in test_df.iterrows():
    graph_A = get_molecular_graph(row['reactant_smiles_A'])
    graph_B = get_molecular_graph(row['reactant_smiles_B'])
    if graph_A is None or graph_B is None:
        continue
    fp_A = get_fingerprint(row['reaction_fingerprint_A'])
    fp_B = get_fingerprint(row['reaction_fingerprint_B'])
    esm_emb_A = get_esm_embedding(row['enzymeA_sequence'], esm_embeddings)
    esm_emb_align_A = get_esm_embedding(row['aligned_enzymeA'], esm_embeddings)
    aligned_seq_A = torch.tensor([ord(aa) for aa in row['aligned_enzymeA']], dtype=torch.long)
    aligned_seq_B = torch.tensor([ord(aa) for aa in row['aligned_enzymeB']], dtype=torch.long)
    val_data.append((graph_A, graph_B, fp_A, fp_B, esm_emb_A, esm_emb_align_A, aligned_seq_A, aligned_seq_B))
    # Add reversed data
    val_data.append((graph_B, graph_A, fp_B, fp_A, esm_emb_A, esm_emb_align_A, aligned_seq_B, aligned_seq_A))

val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Initialize model
model = CATEE(
    gcn_in_channels=9,  # Number of atom features
    gcn_hidden_channels=1024,
    num_gcn_layers=4,
    num_edge_types=4,  # Number of bond types
    esm_dim=1280,  # Dimension of ESM embeddings
    fc_hidden_dim=512
)

# Train the model
train_catee(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4)

# Save the model
torch.save(model.state_dict(), 'catee_model.pt')
