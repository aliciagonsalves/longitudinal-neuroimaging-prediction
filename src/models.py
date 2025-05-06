# src/models.py

import torch
import torch.nn as nn

class BaselineLSTMRegressor(nn.Module):
    """
    A baseline LSTM model for sequence regression (e.g., predicting next CDR).
    Takes sequences of clinical + pre-computed MRI features as input.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        """
        Args:
            input_size (int): The number of features per time step in the input sequence.
                              (Should match the last dimension of sequences_padded from DataLoader).
            hidden_size (int): The number of features in the LSTM hidden state.
            num_layers (int): The number of recurrent layers in the LSTM.
            dropout_prob (float): Dropout probability for the Dropout layer applied after LSTM.
                                  Also used internally by LSTM if num_layers > 1.
        """
        super().__init__() # Initialize the parent nn.Module class

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Define the LSTM layer
        # batch_first=True expects input shape (batch_size, seq_len, input_size)
        # dropout applies dropout between LSTM layers if num_layers > 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0 # Only apply LSTM internal dropout if layers > 1
        )

        # Define a Dropout layer to be applied AFTER the LSTM output
        # This is important for regularization and for enabling MC Dropout during inference
        self.dropout = nn.Dropout(p=dropout_prob)

        # Define the final fully connected layer (Linear layer)
        # It maps the final LSTM hidden state to the single output value (predicted CDR)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, sequences_padded, lengths=None):
        """
        Defines the forward pass of the model.

        Args:
            sequences_padded (Tensor): Batch of padded input sequences.
                                       Shape: (batch_size, max_seq_len, input_size)
            lengths (Tensor, optional): Tensor containing the original lengths of sequences.
                                        Shape: (batch_size,). Not strictly needed for this
                                        simple forward pass but good practice to accept.

        Returns:
            Tensor: The predicted output value for each sequence in the batch. Shape: (batch_size, 1)
        """
        # LSTM processing
        # Output contains hidden states for ALL time steps (batch, seq_len, hidden_size)
        # hidden is a tuple (h_n, c_n):
        # h_n contains the FINAL hidden state for each element in batch (num_layers, batch, hidden_size)
        # c_n contains the FINAL cell state for each element in batch (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(sequences_padded)

        # We need the hidden state from the last layer for the final prediction.
        # h_n shape is (num_layers, batch_size, hidden_size)
        # We take the hidden state from the last layer, h_n[-1] -> shape (batch_size, hidden_size)
        last_hidden_state = h_n[-1]

        # Apply dropout to the final hidden state
        dropout_output = self.dropout(last_hidden_state)

        # Pass the dropout-applied hidden state through the final linear layer
        prediction = self.fc(dropout_output)

        # Prediction shape should be (batch_size, 1)
        return prediction