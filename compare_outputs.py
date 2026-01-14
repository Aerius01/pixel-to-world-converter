#!/usr/bin/env python3
"""
Compare two GPS trajectory output CSV files for equality.

Usage:
    python compare_outputs.py <output_file> <reference_file>
    python compare_outputs.py <output_file> <reference_file> --tolerance 1e-6
"""

import sys
import argparse
import pandas as pd
import numpy as np


def compare_csv_files(output_file, reference_file, tolerance=1e-15):
    """
    Compare two CSV files numerically and report differences.
    
    Args:
        output_file (str): Path to output CSV file to test
        reference_file (str): Path to reference CSV file (ground truth)
        tolerance (float): Maximum allowed difference for numerical columns
        
    Returns:
        bool: True if files match within tolerance, False otherwise
    """
    print(f"Comparing files:")
    print(f"  Output:    {output_file}")
    print(f"  Reference: {reference_file}")
    print(f"  Tolerance: {tolerance}\n")
    
    # Load files
    try:
        df_output = pd.read_csv(output_file)
        df_reference = pd.read_csv(reference_file)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return False
    
    # Check shapes
    print(f"Shape comparison:")
    print(f"  Output:    {df_output.shape}")
    print(f"  Reference: {df_reference.shape}")
    
    if df_output.shape != df_reference.shape:
        print("❌ MISMATCH: Different shapes!\n")
        return False
    print("✓ Shapes match\n")
    
    # Check columns
    print(f"Column comparison:")
    output_cols = list(df_output.columns)
    reference_cols = list(df_reference.columns)
    
    if output_cols != reference_cols:
        print(f"  Output columns:    {output_cols}")
        print(f"  Reference columns: {reference_cols}")
        print("❌ MISMATCH: Different columns!\n")
        return False
    print(f"✓ All {len(output_cols)} columns match\n")
    
    # Compare numerical columns
    numeric_cols = df_output.select_dtypes(include=[np.number]).columns
    all_match = True
    
    if len(numeric_cols) > 0:
        print(f"Comparing {len(numeric_cols)} numerical column(s):")
        for col in numeric_cols:
            diff = np.abs(df_output[col] - df_reference[col])
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            if max_diff > tolerance:
                print(f"  ❌ {col:20s}: max_diff={max_diff:.10e}, mean_diff={mean_diff:.10e}")
                all_match = False
            else:
                print(f"  ✓ {col:20s}: max_diff={max_diff:.10e}, mean_diff={mean_diff:.10e}")
        print()
    
    # Compare string/object columns
    string_cols = df_output.select_dtypes(include=['object']).columns
    
    if len(string_cols) > 0:
        print(f"Comparing {len(string_cols)} string column(s):")
        for col in string_cols:
            if df_output[col].equals(df_reference[col]):
                print(f"  ✓ {col:20s}: identical")
            else:
                num_diff = (df_output[col] != df_reference[col]).sum()
                print(f"  ❌ {col:20s}: {num_diff} rows differ")
                all_match = False
        print()
    
    # Final result
    if all_match:
        print("="*60)
        print("✅ SUCCESS: Files match within tolerance!")
        print("="*60)
        return True
    else:
        print("="*60)
        print("❌ FAILURE: Files have differences!")
        print("="*60)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compare two CSV trajectory output files for equality.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with default tolerance (1e-15)
  %(prog)s output.csv reference.csv
  
  # Compare with custom tolerance
  %(prog)s output.csv reference.csv --tolerance 1e-6
  
  # Using with test automation
  %(prog)s /tmp/test_output.csv expected_output.csv && echo "Test passed"
        """
    )
    
    parser.add_argument(
        'output_file',
        help='Path to output CSV file to test'
    )
    parser.add_argument(
        'reference_file',
        help='Path to reference CSV file (ground truth)'
    )
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=1e-15,
        metavar='TOL',
        help='Maximum allowed difference for numerical columns (default: 1e-15)'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    match = compare_csv_files(args.output_file, args.reference_file, args.tolerance)
    
    # Exit with appropriate code
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
