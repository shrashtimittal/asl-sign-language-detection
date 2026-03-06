import matplotlib.pyplot as plt
import json

# -------------------------------------------------------------------
# 1) Replace these lists with the values you printed during training.
#    You can manually copy the epoch-wise accuracies and losses from
#    your PowerShell/VSCode output into these arrays.
# -------------------------------------------------------------------
train_acc = [0.7744, 0.8925, 0.9899, 0.9970, 0.9976, 0.9982, 0.9985]
val_acc   = [0.9259, 0.9394, 0.9947, 0.9998, 0.9999, 0.9994, 0.9999]
train_loss = [0.65, 0.42, 0.08, 0.03, 0.02, 0.015, 0.012]   # example
val_loss   = [0.30, 0.24, 0.02, 0.01, 0.009, 0.008, 0.008]  # example

epochs = range(1, len(train_acc) + 1)

# Accuracy plot
plt.figure(figsize=(8,6))
plt.plot(epochs, train_acc, 'o-', label='Training Accuracy')
plt.plot(epochs, val_acc, 's-', label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(r"E:\ASL_Project\results\accuracy_curve.png", dpi=200)
plt.show()

# Loss plot
plt.figure(figsize=(8,6))
plt.plot(epochs, train_loss, 'o-', label='Training Loss')
plt.plot(epochs, val_loss, 's-', label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.savefig(r"E:\ASL_Project\results\loss_curve.png", dpi=200)
plt.show()
