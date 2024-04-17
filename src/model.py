import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import lightning as L
import torchvision


class NN(L.LightningModule):
  def __init__(self, inputsize, learning_rate, num_classes):
    super().__init__()
    self.lr = learning_rate
    self.l1 = nn.Linear(inputsize, 128)
    self.l2 = nn.Linear(128, num_classes)
    self.loss_fn = nn.CrossEntropyLoss()
    self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
    self.training_step_outputs = []

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    return x
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    loss, scores, y = self._common_step(batch, batch_idx)
    accuracy = self.accuracy(scores, y)
    f1_score = self.f1_score(scores, y)
    self.training_step_outputs.append(loss)
    self.log_dict({'train_loss': loss,
                   'train_accuracy': accuracy,
                   'train_f1_score': f1_score},
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    
    if batch_idx % 100 == 0:
      x = x[:8]
      grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
      self.logger.experiment.add_image('MNIST_images', grid, self.global_step)
    
    return {"loss": loss, "preds": scores, "target": y}
  
  def on_train_epoch_end(self):
    avg_loss = torch.stack(self.training_step_outputs).mean()
    self.log('train_loss_mean', avg_loss)
    # free up the memory
    self.training_step_outputs.clear()

  
  def validation_step(self, batch, batch_idx):
    loss, scores, y = self._common_step(batch, batch_idx)
    accuracy = self.accuracy(scores, y)
    f1_score = self.f1_score(scores, y)
    self.log_dict({'val_loss': loss,
                   'val_accuracy': accuracy,
                   'val_f1_score': f1_score},
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, scores, y = self._common_step(batch, batch_idx)
    accuracy = self.accuracy(scores, y)
    f1_score = self.f1_score(scores, y)
    self.log_dict({'test_loss': loss,
                   'test_accuracy': accuracy,
                   'test_f1_score': f1_score},
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch
    x = x.reshape(x.size(0), -1)
    scores = self(x) # this is equal to self.forward(x)
    loss = self.loss_fn(scores, y)
    return loss, scores, y
  
  def predict_step(self, batch, batch_idx):
    x, y = batch
    x = x.reshape(x.size(0), -1)
    scores = self(x)
    preds = torch.argmax(scores, dim=1)
    return preds
  
  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)
