import pandas as pd
import torch


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
        member = self.data[self.ID_col].iloc[idx]
        df1 = self.data[self.data[self.ID_col] == member].copy()
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
            ngrams[0] = [self.dx_to_ix['[blank]'] for x in range(self.context_size)]
        elif len(ngrams[0]) < self.context_size:
            size_ = self.context_size - len(ngrams[0])
            for x in range(size_):
                ngrams[0].insert(0, self.dx_to_ix['[blank]'])
        else:
            pass
        ngrams[0] = torch.tensor(ngrams[0]).unsqueeze(0)
        ngrams[1] = torch.tensor(ngrams[1])
        return ngrams