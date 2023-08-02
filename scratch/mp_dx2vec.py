import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

torch.manual_seed(1)


def get_sample_op_claims(sample=1):
    """Returns all outpatient claims for selected sample"""
    df = pd.read_csv(f'https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2008_to_2010_Outpatient_Claims_Sample_{sample}.zip',
                     usecols=['DESYNPUF_ID','CLM_ID','CLM_FROM_DT','PRVDR_NUM','CLM_PMT_AMT',
                               'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4',
                               'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8',
                               'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10', 'ICD9_PRCDR_CD_1',
                               'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3', 'ICD9_PRCDR_CD_4',
                               'ICD9_PRCDR_CD_5', 'ICD9_PRCDR_CD_6', 'ADMTNG_ICD9_DGNS_CD', 'HCPCS_CD_1',
                               'HCPCS_CD_2', 'HCPCS_CD_3', 'HCPCS_CD_4', 'HCPCS_CD_5', 'HCPCS_CD_6',
                               'HCPCS_CD_7', 'HCPCS_CD_8', 'HCPCS_CD_9', 'HCPCS_CD_10', 'HCPCS_CD_11',
                               'HCPCS_CD_12', 'HCPCS_CD_13', 'HCPCS_CD_14', 'HCPCS_CD_15',
                               'HCPCS_CD_16', 'HCPCS_CD_17', 'HCPCS_CD_18', 'HCPCS_CD_19',
                               'HCPCS_CD_20', 'HCPCS_CD_21', 'HCPCS_CD_22', 'HCPCS_CD_23',
                               'HCPCS_CD_24', 'HCPCS_CD_25', 'HCPCS_CD_26', 'HCPCS_CD_27',
                               'HCPCS_CD_28', 'HCPCS_CD_29', 'HCPCS_CD_30', 'HCPCS_CD_31',
                               'HCPCS_CD_32', 'HCPCS_CD_33', 'HCPCS_CD_34', 'HCPCS_CD_35',
                               'HCPCS_CD_36', 'HCPCS_CD_37', 'HCPCS_CD_38', 'HCPCS_CD_39',
                               'HCPCS_CD_40', 'HCPCS_CD_41', 'HCPCS_CD_42', 'HCPCS_CD_43',
                               'HCPCS_CD_44', 'HCPCS_CD_45'],
                     dtype={'CLM_ID':int,'CLM_PMT_AMT':int},
                     compression='zip',
                     engine='c',
                     parse_dates=['CLM_FROM_DT']
                    )
    return df

def get_dx_df(df):
    df_out = df[['DESYNPUF_ID','CLM_FROM_DT',
                 'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4',
                 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8',
                 'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10', 'ICD9_PRCDR_CD_1',
                 'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3', 'ICD9_PRCDR_CD_4',
                 'ICD9_PRCDR_CD_5', 'ICD9_PRCDR_CD_6', 'ADMTNG_ICD9_DGNS_CD',]
               ].melt(id_vars=['DESYNPUF_ID','CLM_FROM_DT'],
                      value_name='dx'
                     ).drop('variable',axis=1)
    df_out = df_out[df_out['dx'].isna()==False]
    df_out = df_out.drop_duplicates(['DESYNPUF_ID','CLM_FROM_DT','dx']).reset_index(drop=True)
    return df_out


class DXDataset(torch.utils.data.Dataset):
    def __init__(self, df, ID_col, context_size, dx_to_ix, mode='CBOW'):
        self.data = df
        self.ID_col = ID_col
        self.context_size=context_size
        self.dx_to_ix = dx_to_ix
        self.mode = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        member = dx_df[self.ID_col].iloc[idx]
        df1 = dx_df[dx_df[self.ID_col] == member].copy()
        self.df1 = df1
        
        if self.mode == 'CBOW':
            if idx >= df1.index.min():
                start = idx - self.context_size//2
            else:
                start = df1.index.min()
            end = idx + self.context_size//2
            
            ngrams_pre = [x for x in df1['dx'].loc[start:(idx-1)]]
            ngrams_post = [x for x in df1['dx'].loc[idx+1:end]]
            ngrams_pre.extend(ngrams_post)
            
            ngrams = [ngrams_pre,
                      df1['dx'].loc[idx]
                     ]
        elif self.mode == 'lead':
            if idx >= df1.index.min():
                start = idx - self.context_size
            else:
                start = df1.index.min()
            
            ngrams = [[x for x in df1['dx'].loc[start:(idx-1)]],
                      df1['dx'].loc[idx]
                     ]
            
            
        if len(ngrams[0]) == 0:
            ngrams[0] = [dx_to_ix['[blank]'] for x in range(self.context_size)]
        elif len(ngrams[0]) < self.context_size:
            size_ = self.context_size - len(ngrams[0])
            for x in range(size_):
                ngrams[0].insert(0, dx_to_ix['[blank]'])
        else:
            pass
        ngrams[0] = torch.tensor(ngrams[0]).unsqueeze(0)
        ngrams[1] = torch.tensor(ngrams[1])
        return ngrams

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


def training_loop(epochs, training_generator, model, loss_function, device, verbose=True):
    losses = []
    model = model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for i, data in enumerate(training_generator):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.zero_grad()
            outputs = model(inputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        if verbose == True:
            if epoch % 1 == 0:
                print(f"Epoch: {epoch} --- Loss: {round(total_loss, 2)}")
    return model, losses


if __name__ == "__main__":
    df = get_sample_op_claims(sample=1)
    dx_df = get_dx_df(df)
    dx_df = dx_df.head(2000)
    vocab = set(dx_df['dx'])
    vocab.add('[blank]')
    dx_to_ix = {dx: i for i, dx in enumerate(vocab)}
    dx_df['dx'] = [dx_to_ix[x] for x in dx_df['dx']]
    print("Data loaded")
    
    # Model parameters
    device = torch.device("mps")
    vocab_size = len(vocab)
    EMBEDDING_DIM = 100
    CONTEXT_SIZE = 10

    loss_function = nn.NLLLoss()
    model = NGramICDModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    #model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)


    data = DXDataset(df=dx_df,
                     ID_col='DESYNPUF_ID',
                     context_size=CONTEXT_SIZE,
                     dx_to_ix=dx_to_ix,
                     mode='CBOW')

    training_generator = torch.utils.data.DataLoader(data,
                                                     batch_size=64,
                                                     shuffle=True,
                                                     num_workers=0
                                                    )
    print("Training model")
    model, losses = training_loop(10,
                                training_generator,
                                model,
                                loss_function,
                                device=device)
    print("Training complete")

    plt.plot(losses)
    plt.show()