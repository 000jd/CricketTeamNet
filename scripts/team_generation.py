import torch
from models.cricket_team_net import CricketTeamNetRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 18
hidden_size = 128
output_size = 11

model = CricketTeamNetRNN(input_size, hidden_size, output_size) 
model.load_state_dict(torch.load('models/cricket_team_model.pth'))
model.eval()

def generate_team_rnn(input_features, all_players):
    with torch.no_grad():
        input_tensor = torch.tensor(input_features, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(1)
        output = model(input_tensor)
        _, indices = torch.topk(output, k=11)

        selected_players = [all_players[i] for i in indices[0]]

    return selected_players