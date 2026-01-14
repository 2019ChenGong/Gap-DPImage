#!/usr/bin/env python3
import subprocess
import os
import sys
from datetime import datetime

# Configuration
DATASETS = ['mnist', 'cifar10', 'octmnist', 'celeba_male', 'camelyon']
MODELS = [
    'stable-diffusion-2-1-base',
    'stable-diffusion-v1-5',
    'stable-diffusion-v1-4',
    'stable-diffusion-2-base',
    'realistic-vision-v5.1',
    'realistic-vision-v6.0',
    'prompt2med',
    ]

METRIC = 'DPFID'
OUTPUT_DIR = 'exp'
OUTPUT_FILE = 'all_dpfid_dpclipping10_result.txt'

def run_dpfid_for_combination(dataset_name, model_name):
    """Run DP-FID evaluation for a single dataset-model combination."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name} | Model: {model_name}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        'python', 'run_metric.py',
        '-m', METRIC,
        '-pm', model_name,
        '-sd', dataset_name,
    ]

    print(f"Running command: {' '.join(cmd)}\n")

    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )

        output = result.stdout
        return_code = result.returncode

        # Print output to console
        print(output)

        if return_code == 0:
            print(f"\n✅ Successfully completed: {dataset_name} + {model_name}")
            status = "SUCCESS"
        else:
            print(f"\n❌ Error occurred: {dataset_name} + {model_name} (exit code: {return_code})")
            status = "FAILED"

        # Try to extract DP-FID score from output
        dpfid_score = None
        for line in output.split('\n'):
            if 'DP-FID Score:' in line:
                try:
                    dpfid_score = float(line.split(':')[-1].strip())
                except:
                    pass

        return {
            'dataset': dataset_name,
            'model': model_name,
            'status': status,
            'return_code': return_code,
            'dpfid_score': dpfid_score,
            'output': output
        }

    except Exception as e:
        error_msg = f"Exception occurred: {str(e)}"
        print(f"\n❌ {error_msg}")
        return {
            'dataset': dataset_name,
            'model': model_name,
            'status': "ERROR",
            'return_code': -1,
            'dpfid_score': None,
            'output': error_msg
        }

def save_results(results, output_path):
    """Save all results to a text file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write(f"DP-FID Evaluation Results\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Write summary table
        f.write("SUMMARY TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Dataset':<15} {'Model':<35} {'Status':<10} {'DP-FID Score':>15}\n")
        f.write("-"*80 + "\n")
        for result in results:
            status_symbol = "✅" if result['status'] == "SUCCESS" else "❌"
            score_str = f"{result['dpfid_score']:.4f}" if result['dpfid_score'] is not None else "N/A"
            f.write(f"{result['dataset']:<15} {result['model']:<35} {status_symbol} {result['status']:<9} {score_str:>15}\n")
        f.write("\n\n")

        # Group results by dataset
        f.write("RESULTS BY DATASET\n")
        f.write("="*80 + "\n\n")
        for dataset in DATASETS:
            f.write(f"Dataset: {dataset}\n")
            f.write("-"*80 + "\n")
            dataset_results = [r for r in results if r['dataset'] == dataset]
            for result in dataset_results:
                score_str = f"{result['dpfid_score']:.4f}" if result['dpfid_score'] is not None else "N/A"
                status_symbol = "✅" if result['status'] == "SUCCESS" else "❌"
                f.write(f"  {status_symbol} {result['model']:<35} DP-FID: {score_str:>10}\n")
            f.write("\n")

        # Write detailed results for each combination
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED OUTPUT FOR EACH COMBINATION\n")
        f.write("="*80 + "\n\n")

        for result in results:
            f.write("="*80 + "\n")
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"Model: {result['model']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Exit Code: {result['return_code']}\n")
            if result['dpfid_score'] is not None:
                f.write(f"DP-FID Score: {result['dpfid_score']:.4f}\n")
            f.write("="*80 + "\n")
            f.write(result['output'])
            f.write("\n\n")

    print(f"\n✅ Results saved to: {output_path}")

def main():
    """Main execution function."""
    total_combinations = len(DATASETS) * len(MODELS)

    print(f"\n{'='*80}")
    print(f"DP-FID Batch Evaluation Script")
    print(f"{'='*80}")
    print(f"Metric: {METRIC}")
    print(f"Datasets ({len(DATASETS)}): {', '.join(DATASETS)}")
    print(f"Models ({len(MODELS)}): {', '.join(MODELS)}")
    print(f"Total combinations: {total_combinations}")
    print(f"Output file: {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
    print(f"{'='*80}\n")

    # Confirm before starting
    response = input(f"Start evaluation for {total_combinations} combinations? (y/n): ")
    if response.lower() != 'y':
        print("Evaluation cancelled.")
        sys.exit(0)

    # Run evaluation for each dataset-model combination
    results = []
    start_time = datetime.now()

    combination_num = 0
    for dataset in DATASETS:
        for model in MODELS:
            combination_num += 1
            print(f"\n{'='*80}")
            print(f"[{combination_num}/{total_combinations}] Dataset: {dataset}, Model: {model}")
            print(f"{'='*80}")

            result = run_dpfid_for_combination(dataset, model)
            results.append(result)

            # Save intermediate results after each combination (in case of crash)
            output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
            save_results(results, output_path)

    end_time = datetime.now()
    elapsed_time = end_time - start_time

    # Print final summary
    print(f"\n{'='*80}")
    print(f"ALL EVALUATIONS COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_time}")
    print(f"\nResults summary:")

    # Summary by dataset
    for dataset in DATASETS:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        success_count = sum(1 for r in dataset_results if r['status'] == 'SUCCESS')
        print(f"\n  Dataset: {dataset}")
        print(f"    Success: {success_count}/{len(MODELS)}")
        for result in dataset_results:
            status_symbol = "✅" if result['status'] == "SUCCESS" else "❌"
            score_str = f"(DP-FID: {result['dpfid_score']:.4f})" if result['dpfid_score'] is not None else ""
            print(f"      {status_symbol} {result['model']:<35} {result['status']:<10} {score_str}")

    # Count total successes and failures
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    fail_count = len(results) - success_count

    print(f"\n{'='*80}")
    print(f"Overall Success: {success_count}/{total_combinations}")
    print(f"Overall Failed: {fail_count}/{total_combinations}")
    print(f"{'='*80}\n")

    # Exit with appropriate code
    sys.exit(0 if fail_count == 0 else 1)

if __name__ == '__main__':
    main()
