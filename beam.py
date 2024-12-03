import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import GCNConv
from nltk.translate.bleu_score import sentence_bleu
import torch.nn as nn
import torch.nn.functional as F
import evaluation
import GPUtil
import argparse
# ----------------------- Setup Device -----------------------
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ----------------------- Dataset -----------------------
class GraphCaptionDataset(Dataset):
    def __init__(self, json_file):
        self.data = json.load(open(json_file, 'r'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def pad_node_features(self, node_features, max_nodes=6, feature_dim=512):
        padded_nodes = np.zeros((max_nodes, feature_dim))
        num_nodes = min(node_features.shape[0], max_nodes)
        padded_nodes[:num_nodes, :] = node_features[:num_nodes, :]
        return padded_nodes

    def pad_edge_features(self, edge_features, max_nodes=6, feature_dim=512):
        padded_edges = np.zeros((max_nodes, max_nodes, feature_dim))
        num_edges = min(edge_features.shape[0], max_nodes)
        padded_edges[:num_edges, :num_edges, :] = edge_features[:num_edges, :num_edges, :]
        return padded_edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        node_path = os.path.join(args.image_extractions, item["id_path"]).replace("//", "\\")
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
            'node_features': torch.tensor(node_features, dtype=torch.float).to(device),
            'edge_features': torch.tensor(edge_features, dtype=torch.float).to(device),
            'caption': tokens.input_ids.squeeze(0).to(device),
            'attention_mask': tokens.attention_mask.squeeze(0).to(device)
        }

# ----------------------- Model Definition -----------------------
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
        self.edge_fc = nn.Linear(18432, 128)
        self.combined_fc = nn.Linear(384, output_dim)

    def forward(self, node_features, edge_features):
        edge_index = self.get_edge_index(node_features.shape[1]).to(device)
        node_emb = F.relu(self.node_gcn1(node_features, edge_index))
        node_emb = F.relu(self.node_gcn2(node_emb, edge_index))

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

    def forward(self, node_features, edge_features, captions=None, attention_mask=None):
        graph_emb = self.gnn(node_features, edge_features)
        graph_emb = self.fc(graph_emb).unsqueeze(1).repeat(1, attention_mask.size(1), 1)

        outputs = self.bert(inputs_embeds=graph_emb, labels=captions, attention_mask=attention_mask)
        return outputs.loss, outputs.logits

# ----------------------- Beam Search Decoding -----------------------
def beam_search_decode(logits, tokenizer, beam_width=3):
    batch_size, seq_len, vocab_size = logits.size()
    all_best_sequences = []

    for b in range(batch_size):  # Process each example in the batch individually
        beams = [(torch.tensor([], dtype=torch.long).to(device), 0)]  # Start with an empty sequence and score of 0

        for t in range(seq_len):
            new_beams = []
            for seq, score in beams:
                probs = torch.log_softmax(logits[b, t, :], dim=-1)  # Process for batch element `b`
                topk_probs, topk_indices = probs.topk(beam_width)

                for i in range(beam_width):
                    new_seq = torch.cat((seq, topk_indices[i].unsqueeze(0)), dim=0)
                    new_score = score + topk_probs[i].item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        best_seq = beams[0][0]
        all_best_sequences.append(best_seq)

    return all_best_sequences

# ----------------------- Training and Evaluation -----------------------
def evaluate(model, val_loader):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    total_bleu = 0
    gen = {}
    gts = {}

    with torch.no_grad():
        for batch_id, batch in enumerate(val_loader):
            node_features = batch['node_features']
            edge_features = batch['edge_features']
            captions = batch['caption']
            attention_mask = batch['attention_mask']

            _, logits = model(node_features, edge_features, attention_mask=attention_mask)
            predicted_ids_list = beam_search_decode(logits, tokenizer)

            for i, predicted_ids in enumerate(predicted_ids_list):
                predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                true_text = tokenizer.decode(captions[i], skip_special_tokens=True)

                total_bleu += sentence_bleu([true_text.split()], predicted_text.split())
                gen[f'{batch_id}_{i}'] = [predicted_text]
                gts[f'{batch_id}_{i}'] = [true_text]
        scores, _ = evaluation.compute_scores(gts, gen)
        print(scores)
    return scores

def train(model, train_loader, val_loader, num_epochs):
    optimizer = Adam(model.parameters(), lr=1e-4)
    best_rouge = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            node_features = batch['node_features']
            edge_features = batch['edge_features']
            captions = batch['caption']
            attention_mask = batch['attention_mask']

            optimizer.zero_grad()
            loss, _ = model(node_features, edge_features, captions, attention_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

        scores = evaluate(model, val_loader)
        rouge_score = scores['ROUGE']
        if rouge_score >best_rouge :
            best_rouge = rouge_score
            torch.save(model.state_dict(), '/home/akshay/Bert_model/models/beam_model.pth')

# ----------------------- Main -----------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Bert_Beam')
    parser.add_argument('--trains_caption', type=str)
    parser.add_argument('--validation_caption', type=str)
    parser.add_argument('--image_extractions', type=str)
    args = parser.parse_args()
    gnn = SimpleGNN(node_input_dim=512, edge_input_dim=18432, output_dim=768).to(device)
    model = GraphToCaptionModel(gnn, 'bert-base-uncased').to(device)

    train_dataset = GraphCaptionDataset(args.trains_caption)
    val_dataset = GraphCaptionDataset(args.validation_caption)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)

    train(model, train_loader, val_loader, num_epochs=20)

