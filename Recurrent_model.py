import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RecurrentNN(nn.Module):
    def __init__(self,hidden_layer, hidden_dim ):
        super(RecurrentNN, self).__init__()
        self.hidden_layer = hidden_layer
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(12,self.hidden_dim,self.hidden_layer,batch_first = True,nonlinearity = 'tanh')
        self.fc = nn.Linear(self.hidden_dim,1)
        
    def forward(self, x):
        h0 = Variable(torch.zeros(self.hidden_layer, x.size(0), self.hidden_dim))
        out,hn = self.rnn(x,h0)
        out = F.sigmoid(self.fc(out))
        return out

   
