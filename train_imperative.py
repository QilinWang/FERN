import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fern_imperative import FERN, FERNConfig
from data_and_metrics import load_etth2_data
import argparse
import os
import pickle

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load Data
    print("Loading data...")
    data_path = os.path.join(args.data_dir, "ETTh2.csv")
    
    # Ensure dimensions are even for complex operations
    seq_len = args.seq_len + (args.seq_len % 2)
    pred_len = args.pred_len + (args.pred_len % 2)

    try:
        # scale=False matches the paper's methodology for ETT datasets
        train_loader, val_loader, _, global_std, channels = load_etth2_data(
            data_path, seq_len, pred_len, args.batch_size, device, scale=False
        )
    except FileNotFoundError as e:
        print(e)
        return
    print("Data loaded.")

    # 2. Initialize Model Configuration (using Dataclass)
    dim_augment = args.dim_augment
    if dim_augment % 2 != 0: dim_augment += 1

    config = FERNConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        channels=channels,
        device=str(device),
        dim_augment=dim_augment,
        dim_hidden=args.dim_hidden,
        householder_reflects_data=args.num_reflects
    )
    model = FERN(config).to(device)
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 3. Setup Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Using Huber loss as specified in the paper (delta=1.0)
    criterion = nn.HuberLoss(delta=1.0)

    # 4. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            # Handle potential mismatch if pred_len was adjusted
            if batch_y.shape[1] != config.pred_len:
                batch_y = F.pad(batch_y, (0, 0, 0, config.pred_len - batch_y.shape[1]))

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if batch_y.shape[1] != config.pred_len:
                    batch_y = F.pad(batch_y, (0, 0, 0, config.pred_len - batch_y.shape[1]))

                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss (Huber): {val_loss:.4f}")

        # Early Stopping and Checkpointing
        if epoch + 1 > args.grace_period:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model state, config (dataclass), and global_std using pickle
                with open(args.checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'global_std': global_std
                    }, f)
                print(f"Checkpoint saved to {args.checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    break

    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Imperative Training Script for FERN on ETTh2")
    # Data and Paths
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_path', type=str, default='fern_etth2_imperative.pkl')
    # Training Hyperparameters
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=15) # Reduced for minimal example
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--grace_period', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    # Model Hyperparameters
    parser.add_argument('--dim_augment', type=int, default=128)
    parser.add_argument('--dim_hidden', type=int, default=128)
    parser.add_argument('--num_reflects', type=int, default=4)
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    train(args)