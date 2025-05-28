import pandas as pd
import torch
import numpy as np
import os
import pathlib
from esm import FastaBatchedDataset, pretrained
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


def extract_embeddings(model_name, fasta_file, output_dir, tokens_per_batch=4096, seq_length=1022,repr_layers=[33]):
    
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=alphabet.get_batch_converter(seq_length), 
        batch_sampler=batches
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            
            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                
                filename = output_dir / f"{entry_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))

                result = {"entry_id": entry_id}
                result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }

                torch.save(result, filename)

model_name = 'esm2_t33_650M_UR50D'
fasta_file = pathlib.Path('./examples/human_proteome.fasta')
output_dir = pathlib.Path('./results/human_proteome_embeddings')
extract_embeddings(model_name, fasta_file, output_dir)

def load_embeddings_from_directory(directory, layer=33):

    embeddings_list = []
    entry_ids_list = []

    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            filepath = os.path.join(directory, filename)
            embedding_dict = torch.load(filepath)

            embedding = embedding_dict['mean_representations'][layer]
            entry_id = embedding_dict['entry_id']

            embeddings_list.append(embedding)
            entry_ids_list.append(entry_id)

    embeddings_tensor = torch.stack(embeddings_list)
    return embeddings_tensor, entry_ids_list

directory = "./results/human_proteome_embeddings"
embeddings_tensor, entry_ids = load_embeddings_from_directory(directory)

test_dataset = TensorDataset(embeddings_tensor)
batch_size = 16
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class RNN_twoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, n_layers, drop_prob):
        """
        Initialize the model by setting up the layers.
        """
        super(RNN_twoLSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lstm1 = nn.LSTM(input_size, hidden_size_1, n_layers, dropout=drop_prob, batch_first=True)  

        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, n_layers, dropout=drop_prob, batch_first=True) 

        self.fc1 = nn.Linear(hidden_size_2, hidden_size_3)
        self.sig1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size_3, output_size)
        self.sig2 = nn.Sigmoid()

    def forward(self, x):
        
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)

        out = self.fc1(out2)
        out = self.sig1(out)
      
        out = self.fc2(out)
        out = self.sig2(out)
       
        return out

input_size = 1280
output_size = 1
hidden_size_1 = 64 
hidden_size_2 = 128 
hidden_size_3 = 64 
n_layers = 2

drop_prob = 0
batch_size = 1 

model = RNN_twoLSTM(input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, n_layers, drop_prob)
print(model)

model_path = './models/Model'
model.load_state_dict(torch.load(model_path))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  

model.eval()
predictions = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0].to(device)  
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())  

results = list(zip(entry_ids, predictions))

for entry_id, prediction in results:
    print(f"Entry ID: {entry_id}, Prediction: {prediction}")


