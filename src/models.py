# src/models.py
"""
PyTorch model definitions for longitudinal prediction of cognitive decline.
Includes:
    - BaselineLSTMRegressor: An LSTM model for sequences of tabular features.
    - Simple3DCNN: A 3D CNN for extracting features from MRI scans.
    - ModularLateFusionLSTM: A hybrid model combining a 3D CNN for MRI features
      and LSTMs for both MRI-derived and tabular feature sequences,
      followed by late fusion for prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineLSTMRegressor(nn.Module):
    """
    A baseline LSTM model for sequence regression (e.g., predicting next CDR score).
    Takes sequences of tabular clinical/demographic features as input.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout_prob: float):
        """
        Args:
            input_size (int): The number of features per time step in the input sequence.
                              (Should match the last dimension of sequences_padded from DataLoader).
            hidden_size (int): The number of features in the LSTM hidden state.
            num_layers (int): The number of recurrent layers in the LSTM.
            dropout_prob (float): Dropout probability for the nn.Dropout layer applied 
                                  after LSTM hidden states. Also used for internal dropout 
                                  between LSTM layers if num_layers > 1.
        """
        super().__init__() 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Define the LSTM layer
        # batch_first=True expects input shape (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0 # LSTM's internal dropout between layers
        )

        # Define a Dropout layer to be applied to the LSTM's final output hidden state
        self.dropout_layer = nn.Dropout(p=dropout_prob)

        # Define the final fully connected layer (Linear layer)
        # It maps the final LSTM hidden state to the single output value
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, sequences_padded: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Defines the forward pass of the BaselineLSTMRegressor model.

        Args:
            sequences_padded (Tensor): Batch of padded input sequences.
                                       Shape: (batch_size, max_seq_len, input_size).
            lengths (Tensor, optional): Tensor containing the original lengths of sequences
                                        before padding. Shape: (batch_size,).
                                        If provided, used for packing sequences.

        Returns:
            Tensor: The predicted output value for each sequence in the batch. 
                    Shape: (batch_size, 1).
        """
        # LSTM processing
        # If lengths are provided, pack padded sequence for efficiency, then pass to LSTM.
        # Ensure lengths are on CPU for pack_padded_sequence.
        if lengths is not None and sequences_padded.device.type != 'mps': # pack_padded_sequence can be problematic on MPS
            packed_input = nn.utils.rnn.pack_padded_sequence(
                sequences_padded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, (h_n, c_n) = self.lstm(packed_input)
            # h_n contains the FINAL hidden state: (num_layers * num_directions, batch_size, hidden_size)
        else:
            # If no lengths or on MPS, pass the padded sequence directly.
            # LSTM can handle padded sequences, but packing is more efficient for varying lengths.
            _, (h_n, c_n) = self.lstm(sequences_padded)
            # h_n contains the FINAL hidden state: (num_layers * num_directions, batch_size, hidden_size)

        # We need the hidden state from the last layer for the final prediction.
        # h_n[-1] gives the hidden state of the last LSTM layer: (batch_size, hidden_size)
        last_hidden_state = h_n[-1]

        # Apply dropout to the final hidden state before the fully connected layer
        dropout_output = self.dropout_layer(last_hidden_state)

        # Pass the dropout-applied hidden state through the final linear layer
        prediction = self.fc(dropout_output)

        return prediction


class Simple3DCNN(nn.Module):
    """
    A simple 3D CNN designed to extract a feature vector from 3D MRI scans.
    Consists of three convolutional blocks followed by adaptive pooling and a fully connected layer.
    """
    def __init__(self, input_channels: int = 1, cnn_output_features: int = 128):
        """
        Args:
            input_channels (int): Number of channels in the input MRI scan (e.g., 1 for T1w).
            cnn_output_features (int): Number of features in the output vector from the CNN.
        """
        super().__init__()
        self.input_channels = input_channels
        self.cnn_output_features = cnn_output_features

        # Convolutional Block 1
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16) # Batch normalization
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) # Halves spatial dimensions

        # Convolutional Block 2
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Adaptive Average Pooling to ensure a fixed-size output before flattening.
        # Outputs a tensor of shape (Batch, 64_channels, 2, 2, 2) in this configuration.
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2)) 
        self.flatten = nn.Flatten()

        # Calculate the number of features after flattening the output of adaptive_pool
        # For 64 channels and (2,2,2) pooling output: 64 * 2 * 2 * 2 = 512
        self.fc_input_dim = 64 * 2 * 2 * 2

        # Fully Connected Layer to map to the desired number of output features
        self.fc_out = nn.Linear(self.fc_input_dim, cnn_output_features)
        self.relu_fc_out = nn.ReLU() # ReLU after the FC layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Simple3DCNN model.

        Args:
            x (Tensor): Input MRI scans. 
                        Shape: (batch_size_effective, input_channels, D, H, W).
                        'batch_size_effective' could be (actual_batch_size * max_seq_len)
                        if processing all MRI scans from all sequences in a batch simultaneously.
        Returns:
            Tensor: Extracted feature vector. Shape: (batch_size_effective, cnn_output_features).
        """
        # Pass through Convolutional Block 1
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        # Pass through Convolutional Block 2
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # Pass through Convolutional Block 3
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # Adaptive Pooling and Flatten
        x = self.adaptive_pool(x)
        x = self.flatten(x) # Shape: (batch_size_effective, self.fc_input_dim)

        # Final Fully Connected Layer
        x = self.relu_fc_out(self.fc_out(x))
        
        return x


class ModularLateFusionLSTM(nn.Module):
    """
    A hybrid deep learning model for longitudinal data that combines features from
    a 3D CNN (for MRI scans) and tabular clinical data using separate LSTM streams,
    followed by late fusion and a final prediction layer.

    The model is designed to predict a continuous target (e.g., next CDR score).
    It includes an option for modality dropout during training for robustness.
    """
    def __init__(self,
                 cnn_input_channels: int,
                 cnn_output_features: int,
                 tabular_input_size: int,
                 mri_lstm_hidden_size: int,     
                 tabular_lstm_hidden_size: int, 
                 num_lstm_layers: int,
                 lstm_dropout_prob: float,
                 num_classes: int = 1, # For regression, out_features of the final FC layer
                 modality_dropout_rate: float = 0.0 
                ):
        """
        Args:
            cnn_input_channels (int): Number of input channels for the 3D CNN (e.g., 1 for T1w MRI).
            cnn_output_features (int): Number of features output by the 3D CNN feature extractor.
                                       This becomes the input_size for the mri_lstm.
            tabular_input_size (int): Number of features in the tabular data sequence per time step.
            mri_lstm_hidden_size (int): Hidden size for the LSTM processing MRI-derived features.
            tabular_lstm_hidden_size (int): Hidden size for the LSTM processing tabular features.
            num_lstm_layers (int): Number of layers for both LSTMs.
            lstm_dropout_prob (float): Dropout probability for nn.Dropout after LSTMs and for
                                       LSTM's internal dropout (if num_lstm_layers > 1).
            num_classes (int): Number of output classes (1 for regression).
            modality_dropout_rate (float): Probability of dropping out an entire modality stream
                                           (MRI or Tabular) during training. Defaults to 0.0 (no dropout).
        """
        super().__init__()

        self.cnn_feature_extractor = Simple3DCNN(
            input_channels=cnn_input_channels,
            cnn_output_features=cnn_output_features
        )

        # LSTM for sequences of MRI-derived features
        self.mri_lstm = nn.LSTM(
            input_size=cnn_output_features, 
            hidden_size=mri_lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout_prob if num_lstm_layers > 1 else 0
        )

        # LSTM for sequences of tabular features
        self.tabular_lstm = nn.LSTM(
            input_size=tabular_input_size,
            hidden_size=tabular_lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout_prob if num_lstm_layers > 1 else 0
        )

        # Calculate input size for the final fully connected layer after fusion
        fused_features_size = mri_lstm_hidden_size + tabular_lstm_hidden_size

        # Dropout layer applied after concatenating features from both LSTM streams
        self.dropout_after_fusion = nn.Dropout(lstm_dropout_prob) 
        
        # Final fully connected layer for prediction
        self.final_fc = nn.Linear(fused_features_size, num_classes)

        self.modality_dropout_rate = modality_dropout_rate
        if not (0.0 <= self.modality_dropout_rate < 1.0):
            raise ValueError("modality_dropout_rate must be between 0.0 and 1.0 (exclusive of 1.0)")

    def forward(self, 
                sequences_tabular_padded: torch.Tensor, 
                sequences_mri_padded: torch.Tensor, 
                lengths_tabular: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the ModularLateFusionLSTM model.

        Args:
            sequences_tabular_padded (Tensor): Padded batch of tabular feature sequences.
                                               Shape: (batch_size, max_seq_len, num_tabular_features).
            sequences_mri_padded (Tensor): Padded batch of MRI scan sequences.
                                           Shape: (batch_size, max_seq_len, C, D, H, W).
            lengths_tabular (Tensor): Original lengths of sequences (applies to both tabular and MRI
                                      sequences, assuming they are aligned). Shape: (batch_size,).
                                      Expected on CPU for pack_padded_sequence.
        Returns:
            Tensor: The predicted output value(s) for each sequence in the batch.
                    Shape: (batch_size, num_classes).
        """
        batch_size = sequences_tabular_padded.size(0)
        max_seq_len_actual = sequences_tabular_padded.size(1) # Actual max seq len in this batch

        # --- MRI Stream ---
        # 1. Reshape MRI sequences to pass all scans through CNN: (batch_size * max_seq_len, C, D, H, W)
        #    Need to handle actual sequence lengths to avoid processing padding through CNN if possible,
        #    or ensure CNN is robust to padded inputs (e.g. if it sees zero-volumes).
        #    For simplicity here, we reshape all; packing later handles sequence lengths for LSTM.
        C, D, H, W = sequences_mri_padded.shape[2:] # Get C, D, H, W from input
        mri_reshaped_for_cnn = sequences_mri_padded.reshape(batch_size * max_seq_len_actual, C, D, H, W)
        
        # 2. Extract features using CNN
        # Output: (batch_size * max_seq_len, cnn_output_features)
        cnn_features_flat = self.cnn_feature_extractor(mri_reshaped_for_cnn)
        
        # 3. Reshape CNN features back into a sequence for the MRI LSTM
        # Output: (batch_size, max_seq_len, cnn_output_features)
        mri_feature_sequence = cnn_features_flat.view(batch_size, max_seq_len_actual, -1)

        # 4. Process MRI feature sequence through MRI LSTM
        # pack_padded_sequence expects lengths on CPU.
        packed_mri_input = nn.utils.rnn.pack_padded_sequence(
            mri_feature_sequence, lengths_tabular.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n_mri, _) = self.mri_lstm(packed_mri_input)
        # h_n_mri shape: (num_lstm_layers, batch_size, mri_lstm_hidden_size)
        mri_lstm_output = h_n_mri[-1] # Output of the last LSTM layer for MRI stream

        # --- Tabular Stream ---
        # 1. Process tabular sequences through Tabular LSTM
        packed_tabular_input = nn.utils.rnn.pack_padded_sequence(
            sequences_tabular_padded, lengths_tabular.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n_tabular, _) = self.tabular_lstm(packed_tabular_input)
        # h_n_tabular shape: (num_lstm_layers, batch_size, tabular_lstm_hidden_size)
        tabular_lstm_output = h_n_tabular[-1] # Output of the last LSTM layer for tabular stream

        # --- Modality Dropout (Applied only during training) ---
        if self.training and self.modality_dropout_rate > 0:
            # Create masks for dropping modalities
            # Example: Independent dropout for each modality stream
            if torch.rand(1).item() < self.modality_dropout_rate:
                # print("DEBUG: Dropping MRI Stream for a batch") # For debugging, remove for final
                mri_lstm_output = torch.zeros_like(mri_lstm_output)
            
            if torch.rand(1).item() < self.modality_dropout_rate:
                # print("DEBUG: Dropping Tabular Stream for a batch") # For debugging, remove for final
                tabular_lstm_output = torch.zeros_like(tabular_lstm_output)

        # --- Late Fusion ---
        # Concatenate the outputs (last hidden states) from both LSTMs
        fused_features = torch.cat((mri_lstm_output, tabular_lstm_output), dim=1)
        # fused_features shape: (batch_size, mri_lstm_hidden_size + tabular_lstm_hidden_size)

        # --- Final Prediction ---
        fused_features_dropped = self.dropout_after_fusion(fused_features)
        prediction = self.final_fc(fused_features_dropped)
        # prediction shape: (batch_size, num_classes)

        return prediction