import pandas as pd
import torch
import numpy as np
import pathlib
from esm import FastaBatchedDataset, pretrained
import os
from torch.utils.data import DataLoader 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

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

fasta_file = pathlib.Path('data/train.fasta')
output_dir = pathlib.Path('results/embeddings/train_embeddings')
extract_embeddings(model_name, fasta_file, output_dir)

fasta_file = pathlib.Path('data/test.fasta')
output_dir = pathlib.Path('results/embeddings/test_embeddings')
extract_embeddings(model_name, fasta_file, output_dir)

fasta_file = pathlib.Path('data/validation.fasta')
output_dir = pathlib.Path('results/embeddings/val_embeddings')
extract_embeddings(model_name, fasta_file, output_dir)


def load_embeddings_from_directory(directory, representation_key=33):

    embeddings_list = []
    labels_list = []

    def extract_label_from_entry_id(entry_id):
        if entry_id.startswith('POS'):
            return 1
        elif entry_id.startswith('NEG'):
            return 0
        else:
            raise ValueError(f"Unknown label in entry_id: {entry_id}")

    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            embedding_dict = torch.load(os.path.join(directory, filename))
            embedding = embedding_dict['mean_representations'][representation_key]
            embeddings_list.append(embedding)

            entry_id = embedding_dict['entry_id']
            label = extract_label_from_entry_id(entry_id)
            labels_list.append(label)

    embeddings = torch.stack(embeddings_list)
    labels = torch.tensor(labels_list, dtype=torch.float32).reshape(-1, 1)

    return embeddings, labels

embeddings_train, labels_train = load_embeddings_from_directory("results/embeddings/train_embeddings")
embeddings_test, labels_test = load_embeddings_from_directory("results/embeddings/test_embeddings")
embeddings_val, labels_val = load_embeddings_from_directory("results/embeddings/val_embeddings")


def custom_dataset(list_of_tensors, labels):
    dataset = []
    for tensor, label in zip(list_of_tensors, labels):
        dataset.append((tensor, label))
    return dataset

batch_size = 16

training_dataset = custom_dataset(embeddings_train, labels_train)
test_dataset = custom_dataset(embeddings_test, labels_test)
validation_dataset = custom_dataset(embeddings_val, labels_val)

train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

seed_value = 88 

torch.manual_seed(seed_value)

torch.cuda.manual_seed(seed_value)

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
        
        #1st LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, n_layers, dropout=drop_prob, batch_first=True)  

        #2nd LSTM layer
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, n_layers, dropout=drop_prob, batch_first=True) 

        #fully-connected output layer
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

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, window_size=20):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.window_size = window_size
        self.min_validation_loss = float('inf')
        self.validation_losses = [] 

    def early_stop(self, validation_loss):
        self.validation_losses.append(validation_loss)
        if len(self.validation_losses) > self.window_size:
            self.validation_losses.pop(0)
        mean_loss = sum(self.validation_losses) / len(self.validation_losses)
        print(mean_loss)
        if validation_loss > mean_loss + self.min_delta: 
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping")
                return True
        else:  
            self.counter = 0
            print("NOPEE")
        return False


#hyperparameters 
input_size = 1280
output_size = 1
hidden_size_1 = 64 
hidden_size_2 = 128 
hidden_size_3 = 64 
n_layers = 2

drop_prob = 0
batch_size = 16  

model = RNN_twoLSTM(input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, n_layers, drop_prob)
print(model)

early_stopper = EarlyStopper(patience=5, min_delta=0.005, window_size=20)

num_epochs = 2000
learning_rate = 0.00001 
clip = 1


def train_and_validate(model, train_loader, val_loader, num_epochs, learning_rate, clip):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        real_labels_train = np.array(())
        predicted_labels_train = np.array(())

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()

            outputs = model(inputs)
            
            cpu_outputs_train = outputs.detach().cpu().numpy()
            predicted_labels_train = np.append(predicted_labels_train,cpu_outputs_train)

            cpu_real_labels_train = labels.detach().cpu().numpy()
            real_labels_train = np.append(real_labels_train, cpu_real_labels_train)

            loss = criterion(outputs, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        real_labels_val = np.array(())
        predicted_labels_val = np.array(())

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  

                outputs = model(inputs)
                
                cpu_outputs_val = outputs.detach().cpu().numpy()
                predicted_labels_val = np.append(predicted_labels_val,cpu_outputs_val)

                cpu_real_labels_val = labels.detach().cpu().numpy()
                real_labels_val = np.append(real_labels_val,cpu_real_labels_val)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if early_stopper.early_stop(val_loss):
            break

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')  
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return real_labels_train,predicted_labels_train, real_labels_val, predicted_labels_val

real_labels_train, predicted_labels_train, real_labels_val, predicted_labels_val=train_and_validate(model, train_loader, val_loader, num_epochs, learning_rate, clip)

#training
predicted_labels_train = np.rint(predicted_labels_train)
confusion_matrix_train = confusion_matrix(real_labels_train,predicted_labels_train)
print(confusion_matrix_train)

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_train, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


accuracy_training = np.sum(np.diag(confusion_matrix_train)) / np.sum(confusion_matrix_train)
print("Accuracy training:", accuracy_training)


predicted_labels_val = np.rint(predicted_labels_val)
confusion_matrix_val = confusion_matrix(real_labels_val,predicted_labels_val)
print(confusion_matrix_val)


plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_val, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


accuracy_validation = np.sum(np.diag(confusion_matrix_val)) / np.sum(confusion_matrix_val)
print("Accuracy validation:", accuracy_validation)


def test(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  

    criterion = nn.BCELoss()

    test_losses=[]

    model.eval()
    test_loss = 0.0
    real_labels = np.array(())
    predicted_labels = np.array(())
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  

            outputs = model(inputs)
            cpu_outputs= outputs.detach().cpu().numpy()
            predicted_labels = np.append(predicted_labels,cpu_outputs)

            cpu_real_labels = labels.detach().cpu().numpy()
            real_labels = np.append(real_labels,cpu_real_labels)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(test_losses)
    return real_labels,predicted_labels


real_labels,predicted_labels=test(model, test_loader)


predicted_labels_n = np.rint(predicted_labels)

confusion_matrix = confusion_matrix(real_labels,predicted_labels_n)
print(confusion_matrix)


plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
print("Accuracy testing:", accuracy)


fpr, tpr, _ = roc_curve(real_labels, predicted_labels)
roc_auc = roc_auc_score(real_labels, predicted_labels)

plt.figure()
plt.plot(fpr, tpr, color='darkred', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label ='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('Receiver Operating Characteristic', fontsize=12)
plt.legend(loc="lower right", fontsize=10)
plt.show()

print("AUC Score:", roc_auc)