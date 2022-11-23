import torch
import data

x_data = data.x_data
y_data = data.y_data

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2 / len()

print("predict (befor training)", 4, forward(4).item())

for epoch in range(100):
    for x,y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()
    print(f"epoch = {epoch} loss = {l.item()}")
print("predict (after training)", 4, forward(4).item())


