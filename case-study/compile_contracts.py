#!/usr/bin/env python3
"""
Compile Solidity contracts from the 'Contracts' directory and save ABI, bytecode, etc.
Defaults to compiling all .sol files under 'Contracts' (including ContractA/Withdraw.sol and ContractB/Withdraw.sol)
Can optionally target a single file.
Usage:
    python3 compile_contracts.py
    python3 compile_contracts.py Contracts/ContractA/Withdraw.sol
"""

import sys
import json
from pathlib import Path
import solcx

# ----------------- Configuration -----------------
# Default Solidity compiler version 
SOLC_VERSION = "0.8.30"

# Input directory to scan for .sol files (default: 'Contracts')
CONTRACTS_DIR = Path("contracts")
# Output directory where compiled artifacts will be saved
OUTPUT_DIR = Path("compiled_artifacts")
# -------------------------------------------------

def install_required_solc(version: str):
    """Install the required solc version if it is not already present."""
    installed = [str(v) for v in solcx.get_installed_solc_versions()]
    if version not in installed:
        print(f"Installing solc {version}...")
        solcx.install_solc(version)
    solcx.set_solc_version(version)

def compile_file(filepath: Path) -> dict:
    """
    Compile a single Solidity file and return the raw compilation output.
    """
    print(f"Compiling {filepath}...")
    output = solcx.compile_files(
        [str(filepath)],
        output_values=["abi", "bin", "bin-runtime"],
        solc_version=SOLC_VERSION,
        optimize=True
    )
    return output

def save_artifacts(compiled: dict):
    """
    Save the ABI and bytecode for each contract to the output directory.
    The JSON key is usually 'filepath:ContractName'.
    To avoid overwriting contracts with the same name in different folders,
    the output filename is prefixed with the parent directory of the source file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for key, val in compiled.items():
        # Extract file path and contract name from the key
        # Example: 'Contracts/ContractA/Withdraw.sol:Withdraw'
        if ':' in key:
            path_part, contract_name = key.split(':', 1)
        else:
            path_part = key
            contract_name = Path(path_part).stem

        # Build a unique identifier: parent-folder_contract-name
        parent_dir = Path(path_part).parent.name
        if parent_dir and parent_dir != ".":
            unique_name = f"{parent_dir}_{contract_name}"
        else:
            unique_name = contract_name

        artifact = {
            "contractName": contract_name,
            "sourcePath": path_part,
            "abi": val.get("abi", []),
            "bytecode": val.get("bin", ""),
            "deployedBytecode": val.get("bin-runtime", "")
        }

        # Save to a JSON file named after the contract
        out_file = OUTPUT_DIR / f"{unique_name}.json"
        with open(out_file, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"  -> Artifact saved: {out_file}")

def main():
    # Install correct solc if needed
    install_required_solc(SOLC_VERSION)

    # Determine which file(s) to compile
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.exists():
            print(f"Error: File not found: {target}")
            sys.exit(1)
        files = [target]
    else:
        # Recursively find all .sol files inside CONTRACTS_DIR
        if not CONTRACTS_DIR.exists():
            print(f"Error: Directory '{CONTRACTS_DIR}' does not exist.")
            sys.exit(1)
        files = list(CONTRACTS_DIR.rglob("*.sol"))
        if not files:
            print(f"No .sol files found under '{CONTRACTS_DIR}'.")
            sys.exit(0)

    # Compile each file individually
    for fpath in files:
        try:
            compiled = compile_file(fpath)
            save_artifacts(compiled)
        except solcx.exceptions.SolcError as e:
            print(f"Compilation error in {fpath}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {fpath}: {e}")

if __name__ == "__main__":
    main()