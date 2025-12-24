#!/usr/bin/env python3
import subprocess
import os
import sys
from datetime import datetime

# Configuration
DATASETS = ['mnist', 'cifar10', 'octmnist', 'celeba_male', 'camelyon']
METRIC = 'PE-Select'
OUTPUT_DIR = 'exp'
OUTPUT_FILE = 'all_metric_result.txt'

def run_metric_for_dataset(dataset_name):
    """Run metric evaluation for a single dataset."""
    print(f"\n{'='*80}")
    print(f"Starting evaluation for dataset: {dataset_name}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        'python', 'run_metric.py',
        '-m', METRIC,
        '-sd', dataset_name
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
            print(f"\n✅ Successfully completed evaluation for {dataset_name}")
            status = "SUCCESS"
        else:
            print(f"\n❌ Error occurred during evaluation for {dataset_name} (exit code: {return_code})")
            status = "FAILED"

        return {
            'dataset': dataset_name,
            'status': status,
            'return_code': return_code,
            'output': output
        }

    except Exception as e:
        error_msg = f"Exception occurred: {str(e)}"
        print(f"\n❌ {error_msg}")
        return {
            'dataset': dataset_name,
            'status': "ERROR",
            'return_code': -1,
            'output': error_msg
        }

def save_results(results, output_path):
    """Save all results to a text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write(f"Metric Evaluation Results - {METRIC}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Write summary
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        for result in results:
            status_symbol = "✅" if result['status'] == "SUCCESS" else "❌"
            f.write(f"{status_symbol} {result['dataset']:15s} - {result['status']:10s} (exit code: {result['return_code']})\n")
        f.write("\n\n")

        # Write detailed results for each dataset
        for result in results:
            f.write("="*80 + "\n")
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Exit Code: {result['return_code']}\n")
            f.write("="*80 + "\n")
            f.write(result['output'])
            f.write("\n\n")

    print(f"\n✅ Results saved to: {output_path}")

def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print(f"Metric Evaluation Script")
    print(f"{'='*80}")
    print(f"Metric: {METRIC}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Output file: {os.path.join(OUTPUT_DIR, OUTPUT_FILE)}")
    print(f"{'='*80}\n")

    # Confirm before starting
    response = input("Start evaluation? (y/n): ")
    if response.lower() != 'y':
        print("Evaluation cancelled.")
        sys.exit(0)

    # Run evaluation for each dataset
    results = []
    start_time = datetime.now()

    for i, dataset in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] Processing dataset: {dataset}")
        result = run_metric_for_dataset(dataset)
        results.append(result)

    end_time = datetime.now()
    elapsed_time = end_time - start_time

    # Print final summary
    print(f"\n{'='*80}")
    print(f"ALL EVALUATIONS COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_time}")
    print(f"\nResults summary:")
    for result in results:
        status_symbol = "✅" if result['status'] == "SUCCESS" else "❌"
        print(f"  {status_symbol} {result['dataset']:15s} - {result['status']}")

    # Save results to file
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    save_results(results, output_path)

    # Count successes and failures
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    fail_count = len(results) - success_count

    print(f"\n{'='*80}")
    print(f"Success: {success_count}/{len(DATASETS)}")
    print(f"Failed: {fail_count}/{len(DATASETS)}")
    print(f"{'='*80}\n")

    # Exit with appropriate code
    sys.exit(0 if fail_count == 0 else 1)

if __name__ == '__main__':
    main()
