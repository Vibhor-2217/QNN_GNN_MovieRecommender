import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class QuantumLayer(nn.Module):
    """Simulated Quantum Layer using classical operations"""

    def __init__(self, n_qubits: int, n_layers: int = 2):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Quantum-inspired parameters
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 3))  # 3 rotation angles per qubit
        self.phi = nn.Parameter(torch.randn(n_layers, n_qubits, n_qubits))  # Entanglement parameters

        # Initialize parameters
        nn.init.uniform_(self.theta, 0, 2 * np.pi)
        nn.init.uniform_(self.phi, 0, 2 * np.pi)

    def quantum_rotation(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Simulate quantum rotation gates"""
        # Apply rotation-like transformations
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Simulate Pauli-X, Y, Z rotations
        x_rot = x * cos_theta[..., 0] + torch.roll(x, 1, dims = -1) * sin_theta[..., 0]
        y_rot = x_rot * cos_theta[..., 1] + torch.roll(x_rot, -1, dims = -1) * sin_theta[..., 1]
        z_rot = y_rot * cos_theta[..., 2] * torch.exp(1j * sin_theta[..., 2])

        return z_rot.real  # Take real part for classical processing

    def quantum_entanglement(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Simulate quantum entanglement between qubits"""
        batch_size = x.shape[0]
        n_features = x.shape[1]

        # Create entanglement matrix
        entangle_matrix = torch.cos(phi) + 1j * torch.sin(phi)
        entangle_matrix = entangle_matrix.real  # Take real part

        # Apply entanglement (simulated as feature mixing)
        entangled = torch.matmul(x.unsqueeze(1), entangle_matrix).squeeze(1)

        return entangled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum layer"""
        batch_size = x.shape[0]

        for layer in range(self.n_layers):
            # Apply quantum rotations
            x = self.quantum_rotation(x, self.theta[layer])

            # Apply quantum entanglement
            x = self.quantum_entanglement(x, self.phi[layer])

            # Add non-linearity (measurement-like operation)
            x = torch.tanh(x)  # Quantum measurement simulation

        return x


class QuantumNeuralNetwork(nn.Module):
    """Quantum Neural Network for enhanced embeddings"""

    def __init__(self, input_dim: int, hidden_dim: int = 32, n_qubits: int = 16, n_quantum_layers: int = 3):
        super(QuantumNeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical preprocessing
        self.input_layer = nn.Linear(input_dim, n_qubits)

        # Quantum layers
        self.quantum_layers = nn.ModuleList([
            QuantumLayer(n_qubits, n_layers = 2) for _ in range(n_quantum_layers)
        ])

        # Classical post-processing
        self.hidden_layer = nn.Linear(n_qubits, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum neural network"""
        # Classical preprocessing
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)

        # Quantum processing
        for quantum_layer in self.quantum_layers:
            x_quantum = quantum_layer(x)
            x = x + x_quantum  # Residual connection
            x = self.dropout(x)

        # Classical post-processing
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = self.output_layer(x)

        return x


class QuantumRecommender(nn.Module):
    """Standalone Quantum Recommender Component"""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super(QuantumRecommender, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Classical embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Quantum enhancement networks
        self.user_quantum_net = QuantumNeuralNetwork(embedding_dim, hidden_dim = 32, n_qubits = 16)
        self.item_quantum_net = QuantumNeuralNetwork(embedding_dim, hidden_dim = 32, n_qubits = 16)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for quantum recommender"""
        # Get classical embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Apply quantum enhancement
        user_quantum = self.user_quantum_net(user_emb)
        item_quantum = self.item_quantum_net(item_emb)

        # Combine classical and quantum representations
        user_final = user_emb + 0.3 * user_quantum  # Weighted combination
        item_final = item_emb + 0.3 * item_quantum

        # Predict rating
        predictions = (user_final * item_final).sum(dim = 1)

        return predictions

    def get_quantum_user_embedding(self, user_id: torch.Tensor) -> torch.Tensor:
        """Get quantum-enhanced user embedding"""
        user_emb = self.user_embedding(user_id)
        user_quantum = self.user_quantum_net(user_emb)
        return user_emb + 0.3 * user_quantum

    def get_quantum_item_embedding(self, item_id: torch.Tensor) -> torch.Tensor:
        """Get quantum-enhanced item embedding"""
        item_emb = self.item_embedding(item_id)
        item_quantum = self.item_quantum_net(item_emb)
        return item_emb + 0.3 * item_quantum