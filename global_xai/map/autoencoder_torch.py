import torch
import torch.nn as nn
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder for sequence reconstruction"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        cat_vocab_size: int,
        num_layers: int = 2,
    ):
        """
        Args:
            input_dim: Number of features per event (11 in your case)
            hidden_dim: Hidden dimension for LSTM layers
            latent_dim: Dimension of the latent/bottleneck representation
            num_layers: Number of LSTM layers in encoder/decoder
            dropout: Dropout rate for regularization
        """
        super(LSTMAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.cat_vocab_size = cat_vocab_size

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder_output_cat = nn.Linear(hidden_dim, self.cat_vocab_size)
        self.decoder_output_num = nn.Linear(hidden_dim, input_dim-1)

        self.softmax_act = torch.nn.Softmax(dim=2)


    def encode(self, x):
        """Encode input sequence to latent representation"""
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.encoder_lstm(x)
        # Use the last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        # latent = latent.unsqueeze(1).repeat(1, x.shape[1], 1)
        latent = self.encoder_fc(last_hidden)  # (batch_size, latent_dim)
        # latent = latent / (torch.linalg.norm(latent, axis=-1, keepdims=True) + 1e-9)
        return latent

    def decode(self, latent, seq_len):
        """Decode latent representation to sequence"""

        # # Project latent to hidden dimension
        hidden = self.decoder_fc(latent)  # (batch_size, hidden_dim)

        # # Repeat for sequence length
        hidden = hidden.unsqueeze(1).repeat(
            1, seq_len, 1
        )  # (batch_size, seq_len, hidden_dim)

        # LSTM decoding
        lstm_out, _ = self.decoder_lstm(hidden)

        # Output projection
        cat_output = self.decoder_output_cat(lstm_out)  # (batch_size, seq_len, input_dim)
        num_output = self.decoder_output_num(lstm_out)  # (batch_size, seq_len, input_dim)
        return cat_output, num_output

    def forward(self, x):
        """Forward pass through autoencoder"""
        seq_len = x.shape[1]
        latent = self.encode(x)
        reconstructed_cat, reconstructed_num = self.decode(latent, seq_len)
        return reconstructed_cat, reconstructed_num, latent



