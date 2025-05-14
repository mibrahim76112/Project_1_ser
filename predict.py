import torch
from model import TransformerModel
from data_loader import load_data


X_train, X_test, y_train, y_test = load_data()


X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


input_dim = X_test_tensor.shape[-1]
num_classes = y_test_tensor.shape[-1]
seq_len = X_test_tensor.shape[1]

model = TransformerModel(input_dim=input_dim, num_classes=num_classes, seq_len=seq_len)


model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()

with torch.no_grad():
    y_pred = model(X_test_tensor)

y_pred_probs = torch.sigmoid(y_pred)

y_pred_class = (y_pred_probs > 0.5).float()

print("Predicted classes:", y_pred_class.numpy())
