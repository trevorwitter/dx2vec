import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_sample_op_claims, get_dx_df, DXDataset
from model import NGramICDModeler


def training_loop(epochs, training_generator, model, optimizer, loss_function, device, verbose=True):
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
            if epoch % 5 == 0:
                print(f"Epoch: {epoch} --- Loss: {round(total_loss, 2)}")
    return model, losses

def main(device='mps'):
    device = torch.device('mps')
    
    EMBEDDING_DIM = 100
    CONTEXT_SIZE = 10
    loss_function = nn.NLLLoss()
    
    

    df = get_sample_op_claims(sample=1)
    claims_per_member = round(len(df)/len(df['DESYNPUF_ID'].unique()),2)
    print(f"{claims_per_member} claims per member on average")
    
    df = get_dx_df(df)
    #df = df.head(100000)
    vocab = set(df['dx'])
    vocab.add('[blank]')
    dx_to_ix = {dx: i for i, dx in enumerate(vocab)}
    print(f'vocab size: {len(vocab)}')
    df['dx'] = [dx_to_ix[x] for x in df['dx']]
    vocab_size = len(vocab)
    dx_to_ix = {dx: i for i, dx in enumerate(vocab)}

    data = DXDataset(df=df,
                    ID_col='DESYNPUF_ID',
                    context_size=CONTEXT_SIZE,
                    dx_to_ix=dx_to_ix,
                    mode='CBOW')

    training_generator = torch.utils.data.DataLoader(data,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    )

    model = NGramICDModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    model, losses = training_loop(10,
                                training_generator,
                                model,
                                optimizer,
                                loss_function,
                                device=device,
                                verbose=True)

    torch.save(model.state_dict(), f"models/test_icd_cbow_{EMBEDDING_DIM}.pt")




if __name__ == "__main__":
    main()