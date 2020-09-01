import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from MSPs_model import MultilayerPs
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
y_test_2 = y_test.numpy()
        
model = MultilayerPs()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.06)

model.train()
epochs = 50
errors = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred.squeeze(), y_train)
    errors.append(loss.item())
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()

model.eval()
y_pred = model(x_test)
after_train = criterion(y_pred.squeeze(), y_test)
print('Test loss after Training' , after_train.item())
preds = y_pred.detach().numpy()
preds = preds.round()
accuracy_score(y_test,preds)
