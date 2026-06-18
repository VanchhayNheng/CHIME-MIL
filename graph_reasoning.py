import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalGraphReasoning(nn.Module):
    def __init__(self, feature_dim, hidden_dim, k_neighbors=6, dropout=0.25):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.fc_node = nn.Linear(feature_dim, hidden_dim)
        self.fc_neighbor = nn.Linear(feature_dim, hidden_dim)
        self.att_vector = nn.Parameter(torch.empty(1, 2 * hidden_dim))
        nn.init.xavier_uniform_(self.att_vector)
        self.update_gate = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        if feature_dim == hidden_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(feature_dim, hidden_dim)
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def construct_spatial_graph(self, region_coords):
        dist = torch.cdist(region_coords, region_coords)
        num_regions = dist.shape[2]
        k = min(self.k_neighbors + 1, num_regions)
        _, indices = torch.topk(dist, k=k, dim=2, largest=False)
        batch_size, n_regions, _ = region_coords.shape
        adj = torch.zeros(batch_size, n_regions, n_regions, device=region_coords.device)
        batch_idx = torch.arange(batch_size, device=region_coords.device).view(batch_size, 1, 1)
        node_idx = torch.arange(n_regions, device=region_coords.device).view(1, n_regions, 1)
        adj[batch_idx, node_idx, indices] = 1.0
        return adj

    def forward(self, region_feats, region_coords):
        batch_size, n_regions, _ = region_feats.shape
        adj = self.construct_spatial_graph(region_coords)
        h_nodes = self.fc_node(region_feats)
        h_neighbors = self.fc_neighbor(region_feats)
        node_repeat = h_nodes.unsqueeze(2).expand(-1, -1, n_regions, -1)
        neigh_repeat = h_neighbors.unsqueeze(1).expand(-1, n_regions, -1, -1)
        concat_feats = torch.cat([node_repeat, neigh_repeat], dim=-1)
        scores = (concat_feats * self.att_vector).sum(dim=-1)
        scores = F.leaky_relu(scores, 0.2)
        scores = scores.masked_fill(adj == 0, -1e9)
        attn = F.softmax(scores, dim=2)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        neighbor_messages = torch.bmm(attn, h_neighbors)
        h_updated = self.update_gate(neighbor_messages.reshape(batch_size * n_regions, -1), h_nodes.reshape(batch_size * n_regions, -1))
        h_updated = h_updated.view(batch_size, n_regions, -1)
        h_updated = self.norm(h_updated + self.residual(region_feats))
        global_scores = self.global_attention(h_updated)
        region_importance = torch.softmax(global_scores, dim=1)
        slide_embedding = (h_updated * region_importance).sum(dim=1)
        return slide_embedding, region_importance.squeeze(-1), h_updated
