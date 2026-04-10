import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler  # mixed precision training: https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/


# Train loop: iterate over the training dataset and try to converge to optimal parameters
# scaler: GradScaler instance — create once before the epoch loop and pass in each epoch
#         so its scale history persists across epochs. If None, a new one is created per call.
def train_loop(dataloader, model, loss_fn, optimizer, scaler=None):
  device = next(model.parameters()).device
  use_amp = device.type == 'cuda'
  if scaler is None:
    scaler = GradScaler(enabled=use_amp)

  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.train() # Set model to training mode
  total_loss, correct = 0, 0 # track totals across all batches

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # autocast enables float16 on CUDA for the forward pass; is a no-op on CPU
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
      pred = model(X)
      loss = loss_fn(pred, y)

    optimizer.zero_grad()        # reset gradients of model parameters to zero at each iteration since gradients by default add up
    scaler.scale(loss).backward()         # backpropogate the prediction loss -- scaled gradients
    scaler.step(optimizer)       # adjust parameters by gradients collected in the backward pass
    scaler.update()              # update scaler parameters

    total_loss += loss.item()   # accumulate total loss
    correct += (pred.argmax(1) == y).type(torch.float).sum().item() # accumulate correct predictions

    if batch % 20 == 0:
      current = (batch + 1) * len(X)
      print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

  avg_loss = total_loss / num_batches   # average over batches
  accuracy = correct / size             # fraction correct over full dataset
  return avg_loss, accuracy             # return for plotting


# Validation/Test loop: iterate over the test dataset to check if model performance is improving
# Returns (avg_loss, top1_acc, top3_acc)
def test_loop(dataloader, model, loss_fn):
  device = next(model.parameters()).device
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, top1_correct, top3_correct = 0, 0, 0
  model.eval() # Set model to evaluation mode

  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      top1_correct += (pred.argmax(1) == y).sum().item()
      top3_correct += pred.topk(3, dim=1).indices.eq(y.unsqueeze(1)).any(dim=1).sum().item()

  test_loss  /= num_batches
  top1_acc    = top1_correct / size
  top3_acc    = top3_correct / size
  print(f"Val — loss: {test_loss:.4f}  top-1: {top1_acc:.2%}  top-3: {top3_acc:.2%}")
  return test_loss, top1_acc, top3_acc

def plot_history(history, model_name="Model"):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b--', label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   'b-',  label='Val Loss')
    ax1.set_title(f'{model_name} — Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy (top-1, and top-3 if tracked)
    ax2.plot(epochs, history['train_acc'], 'r--', label='Train Top-1')
    ax2.plot(epochs, history['val_acc'],   'r-',  label='Val Top-1')
    if 'val_top3' in history:
        ax2.plot(epochs, history['val_top3'], 'g-', label='Val Top-3')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()