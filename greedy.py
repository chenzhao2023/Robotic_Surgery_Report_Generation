#Importing necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from transformers import BertTokenizer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import GCNConv
from nltk.translate.bleu_score import sentence_bleu
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM
import evaluation
import argparse

#Utilizing GPU to train the model
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#Preparing the dataset

class GraphCaptionDataset(Dataset):
    def __init__(self, json_file):
        self.data = json.load(open(json_file, 'r'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def pad_node_features(self, node_features, max_nodes=5, feature_dim=512):
        # Pad node features to (6, 512)
        padded_nodes = np.zeros((max_nodes, feature_dim))
        num_nodes = min(node_features.shape[0], max_nodes)
        padded_nodes[:num_nodes, :] = node_features[:num_nodes, :]
        return padded_nodes

    def pad_edge_features(self, edge_features, max_nodes=5, feature_dim=512):
        # Pad edge features to (6, 6, 512)
        padded_edges = np.zeros((max_nodes, max_nodes, feature_dim))
        num_edges = min(edge_features.shape[0], max_nodes)
        padded_edges[:num_edges, :num_edges, :] = edge_features[:num_edges, :num_edges, :]
        return padded_edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        node_path = item["id_path"]
        node_path=os.path.join(args.image_extractions,node_path).replace("//","\\")
        edge_path = node_path.replace("node", "edge")
        
        # Load node and edge features
        node_features = np.load(node_path)
        edge_features = np.load(edge_path)
        # Pad node and edge features
        node_features = self.pad_node_features(node_features)
        edge_features = self.pad_edge_features(edge_features)

        # Convert caption to BERT input tokens
        caption = item["caption"]
        tokens = self.tokenizer(caption, return_tensors="pt", padding='max_length', truncation=True, max_length=50)
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float),
            'edge_features': torch.tensor(edge_features, dtype=torch.float),
            'caption': tokens.input_ids.squeeze(0),   # Target for training
            'attention_mask': tokens.attention_mask.squeeze(0)
        }

# Create data loaders



#Defining GCNConv layer

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(input_dim, output_dim)

    def forward(self, x, edge_index):
        return self.gcn(x, edge_index)

#Processing Graph data with Graph neural networks

class SimpleGNN(nn.Module):
    
    def __init__(self, node_input_dim, edge_input_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.node_gcn1 = GCNLayer(node_input_dim, 128)
        self.node_gcn2 = GCNLayer(128, 256)
        # Update the edge_fc layer input dimension
        self.edge_fc = nn.Linear(12800, 128) 

        self.combined_fc = nn.Linear(384, output_dim)  
    
    def forward(self, node_features, edge_features):
        # Node features passed through GCN layers
       
        edge_index = self.get_edge_index(node_features.shape[1]).to(device)  # Dynamically create the edge index for fully connected graph
        node_emb = F.relu(self.node_gcn1(node_features, edge_index)).to(device)  # Shape: [batch_size, num_nodes, 128]
        node_emb = F.relu(self.node_gcn2(node_emb, edge_index)).to(device)  # Shape: [batch_size, num_nodes, 256]

        # Flatten and process edge features
        edge_emb = F.relu(self.edge_fc(edge_features.view(edge_features.shape[0], -1)))  # Shape: [batch_size, 128]

        # Expand edge_emb to match node_emb shape along the node dimension
        edge_emb = edge_emb.unsqueeze(1).repeat(1, node_emb.shape[1], 1)  # Shape: [batch_size, num_nodes, 128]

        # Concatenate node and edge embeddings along the last dimension
        combined_emb = torch.cat((node_emb, edge_emb), dim=-1)  # Shape: [batch_size, num_nodes, 384]
        
        # Aggregate over nodes and apply final fully connected layer
        graph_emb = self.combined_fc(combined_emb.mean(dim=1))  # Shape: [batch_size, output_dim]

        return graph_emb 
   
    def get_edge_index(self, num_nodes, max_nodes=6):

        # Dynamically create edge index based on actual num_nodes
        row = torch.arange(num_nodes).repeat(num_nodes)
        col = torch.arange(num_nodes).repeat_interleave(num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        # Filter out connections to non-existent padded nodes (only use the first num_nodes)
        valid_edge_mask = (row < num_nodes) & (col < num_nodes)
        edge_index = edge_index[:, valid_edge_mask]

        return edge_index

#Passing the embeddings obtained from Graph model to BERT

class GraphToCaptionModel(nn.Module):
    
    def __init__(self, gnn, bert_model):
        super(GraphToCaptionModel, self).__init__()
        self.gnn = gnn
        self.bert = BertForMaskedLM.from_pretrained(bert_model)
        self.fc = nn.Linear(768, 768)  # Adjust GNN output to match BERT hidden size
      
    def forward(self, node_features, edge_features, captions=None, attention_mask=None):
        graph_emb = self.gnn(node_features, edge_features).to(device)
        graph_emb = self.fc(graph_emb).to(device)
        
        # Repeat graph embeddings to match the sequence length (e.g., 50)
        graph_emb = graph_emb.unsqueeze(1).repeat(1, attention_mask.size(1), 1)  # Shape: [batch_size, sequence_length, hidden_size]
        
        # BERT takes the graph embeddings and generates the output captions
        outputs = self.bert(inputs_embeds=graph_emb.to(device), labels=captions, attention_mask=attention_mask)
        return outputs.loss, outputs.logits

#training the model

def train(model, train_loader, val_loader, num_epochs):
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()

    best_rouge = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            # Move data to the appropriate device
            node_features = batch['node_features'].to(device)
            edge_features = batch['edge_features'].to(device)
            captions = batch['caption'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            loss, logits = model(node_features, edge_features, captions, attention_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        # Evaluate the model at the end of each epoch

        scores = evaluate(model, val_loader)
        print(scores)
        rouge_score = scores['ROUGE']

        if rouge_score > best_rouge:
            best_rouge = rouge_score
            torch.save(model.state_dict(), '/home/akshay/Bert_model/models/nobeam_best_model.pth')
            print(f'Model saved with ROUGE score: {rouge_score}')

#Evaluating the model

def evaluate(model, val_loader):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    total_bleu = 0
    gen, gts = {}, {}

    with torch.no_grad():
        for batch_id, batch in enumerate(val_loader):
            node_features = batch['node_features'].to(device)
            edge_features = batch['edge_features'].to(device)
            captions = batch['caption'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            _, logits = model(node_features, edge_features, attention_mask=attention_mask)
            predicted_ids = torch.argmax(logits, dim=-1)

            predicted_text = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            true_text = tokenizer.batch_decode(captions, skip_special_tokens=True)

            for pred, true in zip(predicted_text, true_text):
                total_bleu += sentence_bleu([true.split()], pred.split())

            for i, (true_caption, pred_caption) in enumerate(zip(true_text, predicted_text)):
                gen[f'{batch_id}_{i}'] = [pred_caption]
                gts[f'{batch_id}_{i}'] = [true_caption]
            
            scores, _ = evaluation.compute_scores(gts, gen)

    return scores
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Bert_Beam')
    parser.add_argument('--trains_caption', type=str)
    parser.add_argument('--validation_caption', type=str)
    parser.add_argument('--image_extractions', type=str)
    args = parser.parse_args()
    train_dataset = GraphCaptionDataset(args.trains_caption)
    val_dataset = GraphCaptionDataset(args.validation_caption)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)
    # Initialize model and GNN

    gnn = SimpleGNN(node_input_dim=512, edge_input_dim=512 * 6 * 6, output_dim=768)
    model = GraphToCaptionModel(gnn, 'bert-base-uncased').to(device)

    # Train the model
    train(model, train_loader, val_loader, num_epochs=20)
