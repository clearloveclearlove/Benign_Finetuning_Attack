"""
Online IAGS Sample Selection - Main Script
Date: 2025-11-06
Updated: 2025-01-13 - Added weighted_gce method
"""

import argparse
import os
import json
import torch
from pathlib import Path
from datetime import datetime

from online_gradient.online_iags_framework import (
    OnlineIAGSSelector,
    GradientNormScore,
    CosineAlignmentScore,
    WeightedCosineAlignmentScore,
    load_model_and_data
)


def main():
    parser = argparse.ArgumentParser(
        description='Online IAGS Sample Selection with Layer-wise Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GGE with last 8 layers
  python run_online_iags.py --method gge --model_name /path/to/model \\
    --dataset dolly_dataset --data_path data.jsonl --k 100 --num_layers 8

  # GCE with last 8 layers
  python run_online_iags.py --method gce --model_name /path/to/model \\
    --harmful_grad_file harmful_grad.pt --dataset dolly_dataset \\
    --data_path data.jsonl --k 100 --num_layers 8 --normalize True

  # Weighted GCE with last 8 layers
  python run_online_iags.py --method weighted_gce --model_name /path/to/model \\
    --harmful_grad_file harmful_grad.pt --safe_grad_file safe_grad.pt \\
    --dataset dolly_dataset --data_path data.jsonl --k 100 --num_layers 8
        """
    )

    # Model and data
    parser.add_argument('--method', type=str, required=True,
                        choices=['gge', 'gce', 'weighted_gce'],
                        help='Selection method: gge, gce, or weighted_gce')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Path to pretrained model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset file')
    parser.add_argument('--dataset_name', type=str, default='custom',
                        help='Dataset name for output')

    # GCE specific
    parser.add_argument('--harmful_grad_file', type=str, default=None,
                        help='Path to harmful gradient file (required for GCE and weighted_gce)')

    # Weighted GCE specific
    parser.add_argument('--safe_grad_file', type=str, default=None,
                        help='Path to safe gradient file (required for weighted_gce)')
    parser.add_argument('--weight_harmful', type=float, default=1.0,
                        help='Weight for harmful gradient (default: 1.0)')
    parser.add_argument('--weight_safe', type=float, default=-1.0,
                        help='Weight for safe gradient (default: -1.0)')

    # Selection parameters
    parser.add_argument('--k', type=int, required=True,
                        help='Number of samples to select')
    parser.add_argument('--max_response_length', type=int, default=None,
                        help='Maximum response length for gradient computation')
    parser.add_argument('--normalize', type=str, default='False',
                        help='Normalize gradients (True/False)')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Use last N transformer layers only (None=all layers)')

    # Training config
    parser.add_argument('--batch_size_training', type=int, default=1,
                        help='Batch size (must be 1 for online computation)')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for selected samples')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Parse normalize
    normalize = args.normalize.lower() in ['true', '1', 'yes']

    # Validate
    if args.batch_size_training != 1:
        print("⚠️  Warning: batch_size_training must be 1 for online computation, setting to 1")
        args.batch_size_training = 1

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("=" * 70)
    print(f"Online IAGS Sample Selection")
    print(f"Method: {args.method.upper()}")
    print(f"User: {os.environ.get('USER', 'unknown')}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Load model and data
    print("Loading model and dataset...")
    config = {
        'model_name': args.model_name,
        'dataset': args.dataset,
        'data_path': args.data_path,
        'batch_size_training': args.batch_size_training,
    }
    model, train_dataloader = load_model_and_data(config)
    print(f"✓ Loaded model: {args.model_name}")
    print(f"✓ Loaded dataset: {len(train_dataloader.dataset)} samples")
    print()

    # Initialize score function based on method
    if args.method == 'gge':
        print("Method: Greedy Gradient Ensemble (GGE)")
        print("Score function: ||G||^2")
        print("Marginal gain: ||g_c||^2 + 2<G_S, g_c>")
        print()

        score_function = GradientNormScore()

    elif args.method == 'gce':
        print("Method: Greedy Cosine Ensemble (GCE)")
        print("Score function: cos(G, g_harmful)")

        if args.harmful_grad_file is None:
            raise ValueError("--harmful_grad_file is required for GCE")

        print(f"Loading harmful anchor: {args.harmful_grad_file}")
        harmful_gradient = torch.load(args.harmful_grad_file)
        print(f"✓ Loaded harmful gradient: {harmful_gradient.numel():,} parameters")

        # 🔥 Get selected parameter names if using layer-wise selection
        if args.num_layers is not None:
            selected_param_names = OnlineIAGSSelector.get_selected_parameter_names(
                model, args.num_layers
            )

            score_function = CosineAlignmentScore(
                harmful_gradient=harmful_gradient,
                device='cuda',
                selected_param_names=selected_param_names,
                model=model
            )
        else:
            # Use full gradient
            score_function = CosineAlignmentScore(
                harmful_gradient=harmful_gradient,
                device='cuda'
            )

        print()

    elif args.method == 'weighted_gce':
        print("Method: Weighted Greedy Cosine Ensemble (Weighted-GCE)")
        print(f"Score function: {args.weight_harmful}×cos(G, g_harmful) + {args.weight_safe}×cos(G, g_safe)")

        if args.harmful_grad_file is None:
            raise ValueError("--harmful_grad_file is required for weighted_gce")
        if args.safe_grad_file is None:
            raise ValueError("--safe_grad_file is required for weighted_gce")

        print(f"Loading harmful anchor: {args.harmful_grad_file}")
        harmful_gradient = torch.load(args.harmful_grad_file)
        print(f"✓ Loaded harmful gradient: {harmful_gradient.numel():,} parameters")

        print(f"Loading safe anchor: {args.safe_grad_file}")
        safe_gradient = torch.load(args.safe_grad_file)
        print(f"✓ Loaded safe gradient: {safe_gradient.numel():,} parameters")

        # 🔥 Get selected parameter names if using layer-wise selection
        if args.num_layers is not None:
            selected_param_names = OnlineIAGSSelector.get_selected_parameter_names(
                model, args.num_layers
            )

            score_function = WeightedCosineAlignmentScore(
                harmful_gradient=harmful_gradient,
                safe_gradient=safe_gradient,
                weight_harmful=args.weight_harmful,
                weight_safe=args.weight_safe,
                device='cuda',
                selected_param_names=selected_param_names,
                model=model
            )
        else:
            # Use full gradient
            score_function = WeightedCosineAlignmentScore(
                harmful_gradient=harmful_gradient,
                safe_gradient=safe_gradient,
                weight_harmful=args.weight_harmful,
                weight_safe=args.weight_safe,
                device='cuda'
            )

        print()

    # Initialize selector
    selector = OnlineIAGSSelector(
        model=model,
        train_dataloader=train_dataloader,
        score_function=score_function,
        max_response_length=args.max_response_length,
        normalize=normalize,
        device='cuda',
        num_layers=args.num_layers
    )

    # Run selection
    selected_indices, marginal_gains, stats = selector.select_samples(
        k=args.k,
        verbose=True
    )

    # Save results
    print("Saving results...")

    # Save selected indices
    indices_file = output_dir / f"{args.dataset_name}_online_iags_{args.method}_k{args.k}_indices.json"
    with open(indices_file, 'w') as f:
        json.dump({
            'selected_indices': selected_indices,
            'num_selected': len(selected_indices),
            'method': args.method,
            'k': args.k,
            'num_layers': args.num_layers,
            'normalize': normalize,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    print(f"✓ Saved indices to: {indices_file}")

    # Save marginal gains
    gains_file = output_dir / f"{args.dataset_name}_online_iags_{args.method}_k{args.k}_gains.json"
    with open(gains_file, 'w') as f:
        json.dump({
            'marginal_gains': marginal_gains,
            'num_gains': len(marginal_gains),
        }, f, indent=2)
    print(f"✓ Saved gains to: {gains_file}")

    # Save statistics
    stats_file = output_dir / f"{args.dataset_name}_online_iags_{args.method}_k{args.k}_stats.json"
    stats.update({
        'method': args.method,
        'k': args.k,
        'num_layers': args.num_layers,
        'normalize': normalize,
        'model_name': args.model_name,
        'dataset': args.dataset,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved stats to: {stats_file}")

    # Extract selected samples and save to new file
    print("\nExtracting selected samples...")
    import json as json_lib

    # Load original dataset
    with open(args.data_path, 'r') as f:
        if args.data_path.endswith('.jsonl'):
            original_data = [json_lib.loads(line) for line in f]
        else:
            original_data = json_lib.load(f)

    # Extract selected samples
    selected_samples = [original_data[idx] for idx in selected_indices]

    # Save selected samples
    selected_file = output_dir / f"{args.dataset_name}_top{args.k}.json"
    with open(selected_file, 'w') as f:
        json_lib.dump(selected_samples, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved selected samples to: {selected_file}")

    print()
    print("=" * 70)
    print("✅ Online IAGS Selection Complete!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Selected {len(selected_indices)} samples")
    print()


if __name__ == '__main__':
    main()