#!/usr/bin/env python3
"""
Script to extract contract names from Solidity files (.sol) in a given folder.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import json

def extract_contract_name(file_path: str) -> List[str]:
    """
    Extract contract names from a Solidity file.
    
    Args:
        file_path (str): Path to the Solidity file
        
    Returns:
        List[str]: List of contract names found in the file
    """
    contract_names = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Remove single-line comments
            content = re.sub(r'//.*', '', content)
            # Remove multi-line comments
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            
            patterns = [
                # Standard contract with optional inheritance
                r'\bcontract\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:is\s+[^\{]*?)?\{',
                # Abstract contract with optional inheritance
                r'\babstract\s+contract\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:is\s+[^\{]*?)?\{',
                # Interface with optional inheritance
                r'\binterface\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:is\s+[^\{]*?)?\{',
                # Library
                r'\blibrary\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                contract_names.extend(matches)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
    
    return contract_names

def get_solidity_contracts(folder_path: str) -> List[Tuple[str, List[str]]]:
    """
    Get all Solidity files and their contract names from a folder.
    
    Args:
        folder_path (str): Path to the folder containing Solidity files
        
    Returns:
        List[Tuple[str, List[str]]]: List of tuples (filename, [contract_names])
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a directory")
    
    # Find all .sol files recursively
    sol_files = list(folder.rglob("*.sol"))
    
    if not sol_files:
        print(f"No .sol files found in '{folder_path}'")
        return []
    
    results = []
    
    for sol_file in sol_files:
        contract_names = extract_contract_name(sol_file)
        # Use relative path for cleaner output
        relative_path = sol_file.relative_to(folder)
        results.append((str(relative_path), contract_names))
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Extract contract names from Solidity files in a folder"
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing Solidity files (.sol)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search recursively in subdirectories (default: True)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path for JSON results"
    )

    args = parser.parse_args()
    
    try:
        contracts = get_solidity_contracts(args.folder)
        
        if not contracts:
            if args.json or args.output:
                output_json = []
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(output_json, f, indent=2)
                    print(f"Empty results written to {args.output}")
                else:
                    print("[]")
            else:
                print("No contracts found.")
            return
        
        if args.json or args.output:
            json_output = []
            for filename, contract_names in contracts:
                json_output.append({
                    "file": filename,
                    "contracts": contract_names
                })
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(json_output, f)
                print(f"Results written to {args.output}")
            else:
                print(json.dumps(json_output, indent=2))
        else:
            print(f"\nFound {len(contracts)} Solidity file(s) with contracts:\n")
            
            for filename, contract_names in contracts:
                if contract_names:
                    contracts_str = ", ".join(contract_names)
                    print(f"📄 {filename}:")
                    print(f"   📝 Contracts: {contracts_str}")
                else:
                    print(f"📄 {filename}:")
                    print(f"   ⚠️  No contracts found")
                print()
            
    except Exception as e:
        if args.json or args.output:
            error_output = {"error": str(e)}
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(error_output, f, indent=2)
                print(f"Error written to {args.output}")
            else:
                print(json.dumps(error_output, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()