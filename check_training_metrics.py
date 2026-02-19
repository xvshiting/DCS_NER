import torch
import sys

# Load checkpoint
checkpoint_path = "checkpoints/run2/last/checkpoint.pt"
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint loaded successfully!")
    
    # Extract training metrics if available
    if 'metrics' in checkpoint:
        print("\nTraining metrics from checkpoint:")
        metrics = checkpoint['metrics']
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo metrics found in checkpoint")
    
    # Extract step information
    if 'step' in checkpoint:
        print(f"\nTraining step: {checkpoint['step']}")
    
    # Extract config
    if 'config' in checkpoint:
        print("\nTraining configuration:")
        config = checkpoint['config']
        print(f"  epochs: {config.get('epochs', 'N/A')}")
        print(f"  batch_size: {config.get('batch_size', 'N/A')}")
        print(f"  grad_accum: {config.get('grad_accum', 'N/A')}")
        print(f"  lr: {config.get('lr', 'N/A')}")
        print(f"  model_arch: {config.get('model_arch', 'N/A')}")
        
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    sys.exit(1)