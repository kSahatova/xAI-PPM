import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 


class LSTMAETrainer:
    """Training and evaluation wrapper for LSTM Autoencoder"""
    def __init__(self, model: torch.nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 100, lr: float = 0.001, patience: int = 10):
        """
        Train the autoencoder
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            lr: Learning rate
            patience: Early stopping patience
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader):
                x_cat, x_num, y_cat, y_num = batch
                x_cat, x_num, y_cat, y_num = (
                    x_cat.to(self.device),
                    x_num.to(self.device),
                    y_cat.to(self.device),
                    y_num.to(self.device),
                )
                
                optimizer.zero_grad()
                input = torch.concat([x_cat.float(), x_num.float()], dim=-1)

                reconstructed, _ = self.model(input)
                loss = criterion(reconstructed, input)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader:
                val_loss, _ = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_lstm_ae.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}")
        
        # Load best model if validation was used
        if val_loader:
            self.model.load_state_dict(torch.load('best_lstm_ae.pth'))
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, List[float]]:
        """Evaluate the model on given data"""
        self.model.eval()
        criterion = nn.MSELoss()
        losses = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                x_cat, x_num, y_cat, y_num = batch
                x_cat, x_num, y_cat, y_num = (
                    x_cat.to(self.device),
                    x_num.to(self.device),
                    y_cat.to(self.device),
                    y_num.to(self.device),
                )
                input = torch.concat([x_cat.float(), x_num.float()], dim=-1)
                reconstructed, _ = self.model(input)
                loss = criterion(reconstructed, input)
                total_loss += loss.item()
                losses.append(loss.item())
        
        recon_error = total_loss / len(data_loader)
        return recon_error, losses
    
    def plot_training_history(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()