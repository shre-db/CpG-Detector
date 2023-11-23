import torch
import torch.nn as nn
    

class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, output_size):
        super(CpGPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x, hc=None):
        # Apply normalization along the sequence dimension
        x = self.embedding(x)

        # Initialize hidden state and cell state if not provided
        if hc is None:
            hc = self.init_hidden(x.size(0))

        # LSTM layer
        out, _ = self.lstm(x, hc)

        # Use the output from the last time step for classification
        out = self.classifier(out[:, -1, :])

        return out
    
    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
