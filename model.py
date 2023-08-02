import torch.nn as nn
import torch.nn.functional as F

class NGramICDModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramICDModeler, self).__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        embeds = self.embeddings(inputs).view(batch_size, 1, self.context_size*self.embedding_dim)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=0).view(batch_size, self.vocab_size)
        return log_probs