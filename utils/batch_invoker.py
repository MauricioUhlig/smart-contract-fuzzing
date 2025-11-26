#!/usr/bin/env python3
"""
Enhanced script that also includes annotation data as arguments
"""

import json
import sys
import os
import subprocess
import argparse


def load_json_config(json_file):
    """Load and parse the JSON configuration file"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_file}': {e}")
        sys.exit(1)


def invoke_fuzzer(source_file, contract_name, annotations, extra_args):
    """Invoke the fuzzer with the specified arguments"""
    # Build the command
    cmd = [
        "python3", "fuzzer/main.py",
        "--source", "examples/"+source_file,
        "--contract", contract_name,
        "-r", "results/"+ extra_args[(extra_args.index('--algorithm')+1)]+ '/'+ source_file.replace('/', '-').replace(".sol", '.json')
    ]

    
    # Add extra arguments
    cmd.extend(extra_args)
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        # Execute the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Fuzzer execution completed successfully")
        print("Output:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Fuzzer execution failed with exit code {e.returncode}")
        print("Output:", e.stdout)
        print("Errors:", e.stderr)
        return False
    except FileNotFoundError:
        print("Error: fuzzer/main.py not found")
        return False


def main():
    parser = argparse.ArgumentParser(description='Process JSON configuration and invoke fuzzer')
    parser.add_argument('json_file', help='Path to the JSON configuration file')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, 
                       help='Extra arguments to pass to the fuzzer')
    
    args = parser.parse_args()
    
    # Load JSON configuration
    config = load_json_config(args.json_file)
    
    # Process each file in the configuration
    success_count = 0
    total_count = 0
    
    for file_config in config:
        source_file = file_config.get("file")
        contracts = file_config.get("contracts", [])
        
        if not source_file:
            print("Warning: Missing 'file' field in configuration, skipping...")
            continue
        
        # Process each contract in the file
        for contract_config in contracts:
            contract_name = contract_config.get("name")
            annotations = contract_config.get("annotations", {})
            
            if not contract_name:
                print(f"Warning: Missing 'name' field for contract in {source_file}, skipping...")
                continue
            
            total_count += 1
            print(f"\nProcessing: {source_file} -> {contract_name}")
            print(f"Annotations: {annotations}")
            
            # Invoke the fuzzer
            if invoke_fuzzer(source_file, contract_name, annotations, args.extra_args):
                success_count += 1
    
    # Print summary
    print(f"\nSummary: {success_count}/{total_count} fuzzer executions completed successfully")
    
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()