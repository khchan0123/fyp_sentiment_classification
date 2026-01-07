import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel 

class BertMultiScaleCNN(nn.Module):
    def __init__(self, num_classes, num_filters, kernel_sizes, dropout, freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        embed_dim = 768
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) 
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, input_ids, attention_mask):
        # 1. BERT
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state.permute(0, 2, 1)
        
        # 2. Multi-Scale CNN
        x_convs = [F.relu(conv(x)) for conv in self.convs]
        x_pool = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x_convs]
        
        # 3. Concat & Classify
        x_cat = torch.cat(x_pool, dim=1)
        return self.fc(self.dropout(x_cat))

class BertHybridOptimized(nn.Module):
    def __init__(self, num_classes, num_filters, kernel_size, lstm_hidden, dropout, freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        embed_dim = 768
        
        self.conv = nn.Conv1d(
            in_channels=embed_dim, 
            out_channels=num_filters, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        
        self.lstm = nn.LSTM(
            input_size=num_filters, 
            hidden_size=lstm_hidden, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # 1. BERT
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state
        
        # 2. CNN (Needs [Batch, Channels, Seq])
        x = x.permute(0, 2, 1) 
        x = F.relu(self.conv(x))
        
        # 3. LSTM (Needs [Batch, Seq, Features])
        x = x.permute(0, 2, 1) 
        _, (h_n, _) = self.lstm(x)
        
        # 4. Concat Hidden States
        hidden_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden_cat))