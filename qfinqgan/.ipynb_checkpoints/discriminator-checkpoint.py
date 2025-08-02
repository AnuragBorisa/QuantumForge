import torch 
import torch.nn as nn

class ClassicalDiscriminator(nn.Module):

    def __init__(self,input_dim:int =8,hidden_dim:int =16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.net(x)

    def predict_proba(self,x:torch.Tensor)->torch.Tensor:
        self.eval()
        with torch.no_grad():
             return self.forward(x)

    
        