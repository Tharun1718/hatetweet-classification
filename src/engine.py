import torch

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch.
    """
    # Round predictions to the closest integer (0 or 1)
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def train_one_epoch(model, iterator, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train() # Set to train mode (enables Dropout)
    
    for text, label in iterator:
        text, label = text.to(device), label.to(device)
        
        optimizer.zero_grad() # Clear previous gradients
        predictions = model(text).squeeze(1) # [batch, 1] -> [batch]
        
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)
        
        loss.backward() # Compute gradients
        
        # Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step() # Update weights
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    num_batches = len(iterator)
    if num_batches == 0:
        return 0.0, 0.0
    return epoch_loss / num_batches, epoch_acc / num_batches

def evaluate(model, iterator, criterion, device):
    """
    Evaluates the model on validation/test data.
    """
    epoch_loss = 0
    epoch_acc = 0
    model.eval() # Set to eval mode (disables Dropout)
    
    with torch.no_grad(): # No gradients needed for validation
        for text, label in iterator:
            text, label = text.to(device), label.to(device)

            predictions = model(text).squeeze(1)
            
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    num_batches = len(iterator)
    if num_batches == 0:
        return 0.0, 0.0
    return epoch_loss / num_batches, epoch_acc / num_batches