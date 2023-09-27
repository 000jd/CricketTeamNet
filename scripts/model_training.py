import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.cricket_team_net import CricketTeamNetRNN
from . import players_name
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 18
hidden_size = 128
output_size = 11
EPOCHES = 1000

# Load preprocessed data
preprocessed_train_path = '../data/preprocessed/preprocessed_train.csv'
preprocessed_test_path = '../data/preprocessed/preprocessed_test.csv'

train_df = pd.read_csv(preprocessed_train_path)
test_df = pd.read_csv(preprocessed_test_path)

X_train = train_df.drop('player', axis=1).values
y_train = train_df['player'].values
X_test = test_df.drop('player', axis=1).values
y_test = test_df['player'].values

player_to_id = {player: idx for idx, player in enumerate(set(players_name.all_players))}
y_train_ids = [player_to_id[player] for player in y_train]
y_test_ids = [player_to_id[player] for player in y_test]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_ids, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_ids, dtype=torch.long).to(device)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

unique_players = set(y_train_ids + y_test_ids)
output_size = len(unique_players)


output_size = len(unique_players)
model = CricketTeamNetRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_size_tuple = (1, input_size)
summary(model, input_size=input_size_tuple, device=device)

train_losses = []
num_epochs = EPOCHES

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (i+1) % 100 == 0:
            avg_loss = total_loss / 100
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
            train_losses.append(avg_loss)
            total_loss = 0

plt.figure(figsize=(10, 6))
sns.set_style('darkgrid')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Steps (in hundreds)')
plt.ylabel('Loss')
plt.title('Training Loss over Steps')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'models/cricket_team_model.pth')