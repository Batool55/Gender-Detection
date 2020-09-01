import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

data = pd.read_csv('all_features.csv')
data['Gender'] = data['Gender'].replace(['male','female'],[0,1])
X = data.iloc[:,1:13].values
y = data.iloc[:,13].values
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle=True, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

class RecurrentNN(nn.Module):
    def __init__(self,hidden_layer, hidden_dim ):
        super(RecurrentNN, self).__init__()
        self.hidden_layer = hidden_layer
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(12,hidden_dim,hidden_layer,batch_first = True,nonlinearity = 'tanh')
        self.fc = nn.Linear(hidden_dim,1)
        
    def forward(self, x):
        h0 = Variable(torch.zeros(hidden_layer, x.size(0), hidden_dim))
        out,hn = self.rnn(x,h0)
        out = F.sigmoid(self.fc(out))
        return out
    
hidden_layer = 1
hidden_dim = 400
model = RecurrentNN(hidden_layer,hidden_dim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.08)
        
model.train()
epochs = 100
errors = []
for epoch in range(epochs):
    optimizer.zero_grad()
    x_train  = Variable(x_train.view(-1, 3169, 12))
    y_train = Variable(y_train )
    y_pred = model(x_train)
    loss = criterion(y_pred.squeeze(), y_train)
    errors.append(loss.item())
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()
    
model.eval()
x_test  = Variable(x_test.view(-1, 1359, 12))
y_pred = model(x_test)
after_train = criterion(y_pred.squeeze(), y_test)
print('Test loss after Training' , after_train.item())
preds = y_pred.detach().numpy()
preds = preds.round()
preds = preds.transpose(2,0,1).reshape(-1,)
accuracy_score(y_test,preds)
   
