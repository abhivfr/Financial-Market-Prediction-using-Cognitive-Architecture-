#!/usr/bin/env python
# rename_regime_column.py - Rename market_regime column to regime for evaluation

import pandas as pd
import os
import sys
import argparse

def rename_regime_column(input_file, output_file=None):
    """
    Rename market_regime column to regime for compatibility with evaluation scripts
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None, will modify in-place)
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check if market_regime column exists
    if 'market_regime' in df.columns:
        print(f"Renaming 'market_regime' column to 'regime'...")
        df = df.rename(columns={'market_regime': 'regime'})
        
        # Save the modified dataframe
        if output_file is None:
            output_file = input_file
        
        print(f"Saving modified data to {output_file}...")
        df.to_csv(output_file, index=False)
        print("Done!")
    else:
        if 'regime' in df.columns:
            print("File already has a 'regime' column. No changes needed.")
        else:
            print("Warning: Neither 'market_regime' nor 'regime' column found in the data.")

def main():
    parser = argparse.ArgumentParser(description="Rename market_regime column to regime for evaluation")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", help="Path to output CSV file (if not specified, will modify in-place)")
    
    args = parser.parse_args()
    
    rename_regime_column(args.input, args.output)

if __name__ == "__main__":
    main() 