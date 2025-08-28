import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

# Relational Graph Convolutional Network (RGCN) for molecular graph embedding
class MolecularRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_edge_types):
        super(MolecularRGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations=num_edge_types))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations=num_edge_types))
        self.fc = nn.Linear(hidden_channels, 1024)

    def forward(self, x, edge_index, edge_type):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type))
        return self.fc(x)

# Improved Cross Attention Mechanism with Scaled Dot-Product Attention
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # Project and reshape into multiple heads
        Q = self.Wq(Q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(K).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(V).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and pass through output layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.fc(attention_output)

# Crossly Attended Transformer for Enzyme Engineering (CATEE)
class CATEE(nn.Module):
    def __init__(self, gcn_in_channels, gcn_hidden_channels, num_gcn_layers, num_edge_types, esm_dim, fc_hidden_dim):
        super(CATEE, self).__init__()
        # Shared RGCN for Molecular Graphs
        self.rgcn = MolecularRGCN(gcn_in_channels, gcn_hidden_channels, num_gcn_layers, num_edge_types)
        
        # Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(gcn_hidden_channels + 2048, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1024)
        )
        
        # Cross Attention Mechanisms
        self.cross_att1 = CrossAttention(dim=esm_dim)
        self.cross_att2 = CrossAttention(dim=esm_dim)
        
        # Output Layers for Aligned Sequence Prediction
        self.fc_delete_or_indel = nn.Linear(esm_dim, 2)  # Binary output: delete or indel
        self.fc_vocabulary = nn.Linear(esm_dim, 21)  # Predict amino acid probability

    def forward(self, graph_A, graph_B, fp_A, fp_B, esm_emb_A, esm_emb_align_A, aligned_seq_A, aligned_seq_B):
        # Molecular Graph Embeddings
        rgcn_emb_A = self.rgcn(graph_A.x, graph_A.edge_index, graph_A.edge_attr)
        rgcn_emb_B = self.rgcn(graph_B.x, graph_B.edge_index, graph_B.edge_attr)
        
        # Chemical Embeddings
        chem_emb_A = torch.cat([rgcn_emb_A, fp_A], dim=1)
        chem_emb_B = torch.cat([rgcn_emb_B, fp_B], dim=1)
        chem_A = self.fc1(chem_emb_A)
        chem_B = self.fc1(chem_emb_B)
        chem_emb = chem_B - chem_A
        
        # Cross Attention 1 for Deletion or Indel Prediction
        Q1 = chem_emb.unsqueeze(1)  # Add sequence dimension
        K1 = esm_emb_A
        V1 = esm_emb_A
        enz_emb_1 = self.cross_att1(Q1, K1, V1)
        delete_or_indel_pred = self.fc_delete_or_indel(enz_emb_1)
        
        # Cross Attention 2 for Amino Acid Prediction
        Q2 = chem_emb.unsqueeze(1)
        K2 = esm_emb_align_A
        V2 = esm_emb_align_A
        enz_emb_2 = self.cross_att2(Q2, K2, V2)
        aa_pred = self.fc_vocabulary(enz_emb_2)
        
        return delete_or_indel_pred, aa_pred

# Training Function
def train_catee(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_delete_indel = nn.CrossEntropyLoss()
    criterion_aa = nn.CrossEntropyLoss()
    early_stopping_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            graph_A, graph_B, fp_A, fp_B, esm_emb_A, esm_emb_align_A, aligned_seq_A, aligned_seq_B = data
            
            optimizer.zero_grad()
            delete_or_indel_pred, aa_pred = model(graph_A, graph_B, fp_A, fp_B, esm_emb_A, esm_emb_align_A, aligned_seq_A, aligned_seq_B)
            
            loss1 = criterion_delete_indel(delete_or_indel_pred.view(-1, 2), aligned_seq_A.view(-1))
            loss2 = criterion_aa(aa_pred.view(-1, 21), aligned_seq_B.view(-1))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                graph_A, graph_B, fp_A, fp_B, esm_emb_A, esm_emb_align_A, aligned_seq_A, aligned_seq_B = data
                delete_or_indel_pred, aa_pred = model(graph_A, graph_B, fp_A, fp_B, esm_emb_A, esm_emb_align_A, aligned_seq_A, aligned_seq_B)
                loss1 = criterion_delete_indel(delete_or_indel_pred.view(-1, 2), aligned_seq_A.view(-1))
                loss2 = criterion_aa(aa_pred.view(-1, 21), aligned_seq_B.view(-1))
                val_loss += (loss1 + loss2).item()
        
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")
