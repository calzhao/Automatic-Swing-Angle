
import torch
import torch.nn as nn

# Define your MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=28):
        super(MLP, self).__init__()
        print("MLP input_dim",input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        # print("x",x.shape)
        # x = x.view(x.shape[0],-1)
        x = self.layers(x)
        return x.squeeze()
    
def load_model(model_path, win_len=3):
    model = MLP(input_dim=(2*win_len+1)*4)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    print ("successfully loaded checkpoint from {}".format(model_path))
    return model