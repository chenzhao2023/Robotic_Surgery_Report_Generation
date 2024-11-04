import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from transformers import BertTokenizer
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM
from nltk.translate.bleu_score import sentence_bleu
import evaluation
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GraphCaptionDataset(Dataset):
    def __init__(self, json_file):
        self.data = json.load(open(json_file, 'r'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def pad_node_features(self, node_features, max_nodes=5, feature_dim=512):
        padded_nodes = np.zeros((max_nodes, feature_dim))
        num_nodes = min(node_features.shape[0], max_nodes)
        padded_nodes[:num_nodes, :] = node_features[:num_nodes, :]
        return padded_nodes

    def pad_edge_features(self, edge_features, max_nodes=5, feature_dim=512):
        padded_edges = np.zeros((max_nodes, max_nodes, feature_dim))
        num_edges = min(edge_features.shape[0], max_nodes)
        padded_edges[:num_edges, :num_edges, :] = edge_features[:num_edges, :num_edges, :]
        return padded_edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        node_path = item["id_path"]
        node_path = os.path.join(args.image_extractions, node_path).replace("//", "\\")
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
            'caption': tokens.input_ids.squeeze(0),
            'attention_mask': tokens.attention_mask.squeeze(0)
        }

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(input_dim, output_dim)

    def forward(self, x, edge_index):
        return self.gcn(x, edge_index)

class SimpleGNN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.node_gcn1 = GCNLayer(node_input_dim, 128)
        self.node_gcn2 = GCNLayer(128, 256)
        self.edge_fc = nn.Linear(12800, 128)  # Change 12800 to 18432 to match reshaped edge features
        self.combined_fc = nn.Linear(384, output_dim)

    def forward(self, node_features, edge_features):
        edge_index = self.get_edge_index(node_features.shape[1]).to(device)
        node_emb = F.relu(self.node_gcn1(node_features, edge_index)).to(device)
        node_emb = F.relu(self.node_gcn2(node_emb, edge_index)).to(device)
        edge_emb = F.relu(self.edge_fc(edge_features.view(edge_features.shape[0], -1)))
        edge_emb = edge_emb.unsqueeze(1).repeat(1, node_emb.shape[1], 1)
        combined_emb = torch.cat((node_emb, edge_emb), dim=-1)
        graph_emb = self.combined_fc(combined_emb.mean(dim=1))
        return graph_emb

    def get_edge_index(self, num_nodes, max_nodes=6):
        row = torch.arange(num_nodes).repeat(num_nodes)
        col = torch.arange(num_nodes).repeat_interleave(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        valid_edge_mask = (row < num_nodes) & (col < num_nodes)
        edge_index = edge_index[:, valid_edge_mask]
        return edge_index

class GraphToCaptionModel(nn.Module):
    def __init__(self, gnn, bert_model):
        super(GraphToCaptionModel, self).__init__()
        self.gnn = gnn
        self.bert = BertForMaskedLM.from_pretrained(bert_model)
        self.fc = nn.Linear(768, 768)
      
    def forward(self, node_features, edge_features, attention_mask=None):
        graph_emb = self.gnn(node_features, edge_features).to(device)
        graph_emb = self.fc(graph_emb).to(device)
        graph_emb = graph_emb.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
        outputs = self.bert(inputs_embeds=graph_emb.to(device), attention_mask=attention_mask)
        return outputs.logits

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

            logits = model(node_features, edge_features, attention_mask=attention_mask)
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
    parser.add_argument('--validation_caption', type=str)
    parser.add_argument('--image_extractions', type=str)
    parser.add_argument('--model_weights', type=str)
    args = parser.parse_args()
    # Load model weights and prepare validation data
    gnn = SimpleGNN(node_input_dim=512, edge_input_dim=512 * 6 * 6, output_dim=768)
    model = GraphToCaptionModel(gnn, 'bert-base-uncased').to(device)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))

    # Set up validation dataset and data loader
    val_dataset = GraphCaptionDataset(args.validation_caption)
    val_loader = DataLoader(val_dataset, batch_size=10, pin_memory=True)

    # Evaluate the model
    scores = evaluate(model, val_loader)
    print("Evaluation Scores:", scores)
