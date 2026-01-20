import numpy as np
from typing import Tuple, Optional
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def compute_reconstruction_errors(model: torch.nn.Module, 
                                  data_loader: DataLoader,
                                  device: Optional[str] = None,
                                  ) -> np.ndarray:
        """
        Compute reconstruction errors for OOD detection
        
        Returns:
            Array of reconstruction errors (one per trace)
        """
        device  = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                reconstructed, _ = model(batch)
                
                # Compute MSE per trace
                mse = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
                errors.extend(mse.cpu().numpy())
        
        return np.array(errors)


def detect_ood(model: torch.nn.Module, traces: np.ndarray, percentile: float = 95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect out-of-distribution traces
    
    Args:
        traces: Array of traces to check
    
    Returns:
        Tuple of (is_ood_array, reconstruction_errors)
    """
    # TODO : correct the ood detection  
    # Compute errors
    dataset = TensorDataset(torch.tensor(traces, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    errors = compute_reconstruction_errors(model, loader)
    threshold = np.percentile(errors, percentile)
    
    # Determine OOD
    is_ood = errors > threshold
    
    return is_ood, errors