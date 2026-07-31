DivFizz
===========

A diversity-based hybrid fuzzer for Ethereum smart contracts, based on Confuzzis.

## Custom Docker image build

```
docker build -t divfuzz .
docker run -it confuzzius:latest
```

## Installation Instructions

### 1. Install Requirements

#### 1.1 Solidity Compiler

``` shell
sudo add-apt-repository ppa:ethereum/ethereum
sudo apt-get update
sudo apt-get install solc
```

#### 1.2 Z3 Prover

Download the [source code of version z3-4.8.5](https://github.com/Z3Prover/z3/releases/tag/Z3-4.8.5)

Install z3 using Python bindings

``` shell
python scripts/mk_make.py --python
cd build
make
sudo make install
```

### 2. Install Fuzzer

``` shell
cd fuzzer
pip install -r requirements.txt
```

## Running Instructions

```
 ██████████    ███              ███████████                                 
░░███░░░░███  ░░░              ░░███░░░░░░█                                 
 ░███   ░░███ ████  █████ █████ ░███   █ ░  █████ ████  █████████  █████████
 ░███    ░███░░███ ░░███ ░░███  ░███████   ░░███ ░███  ░█░░░░███  ░█░░░░███ 
 ░███    ░███ ░███  ░███  ░███  ░███░░░█    ░███ ░███  ░   ███░   ░   ███░  
 ░███    ███  ░███  ░░███ ███   ░███  ░     ░███ ░███    ███░   █   ███░   █
 ██████████   █████  ░░█████    █████       ░░████████  █████████  █████████
░░░░░░░░░░   ░░░░░    ░░░░░    ░░░░░         ░░░░░░░░  ░░░░░░░░░  ░░░░░░░░░ 

usage: main.py [-h] (-s SOURCE | -a ABI) [-c CONTRACT] [-b BLOCKCHAIN_STATE]
               [--solc SOLC_VERSION] [--evm EVM_VERSION]
               [--algorithm {confuzzius,divfuzz}]
               [-g GENERATIONS | -t GLOBAL_TIMEOUT]
               [-n POPULATION_SIZE] [-pc PROBABILITY_CROSSOVER]
               [-pm PROBABILITY_MUTATION]
               [-dw DATA_DEPENDENCY_WEIGHT] [-bw BRANCH_WEIGHT]
               [-uw UNIQUE_BRANCH_WEIGHT]
               [-r RESULTS] [--tag TAG] [--seed SEED] [--cfg]
               [--rpc-host RPC_HOST] [--rpc-port RPC_PORT]
               [--data-dependency DATA_DEPENDENCY]
               [--diversity DIVERSITY]
               [--constraint-solving CONSTRAINT_SOLVING]
               [--environmental-instrumentation ENVIRONMENTAL_INSTRUMENTATION]
               [--max-individual-length MAX_INDIVIDUAL_LENGTH]
               [--max-symbolic-execution MAX_SYMBOLIC_EXECUTION]
               [-v]

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE, --source SOURCE
                        Solidity smart contract source code file (.sol).
  -a ABI, --abi ABI     Smart contract ABI file (.json).
  -c CONTRACT, --contract CONTRACT
                        Contract name to be fuzzed (if Solidity source code
                        file provided) or blockchain contract address (if ABI
                        file provided).
  -b BLOCKCHAIN_STATE, --blockchain-state BLOCKCHAIN_STATE
                        Initialize fuzzer with a blockchain state by providing
                        a JSON file (if Solidity source code file provided) or
                        a block number (if ABI file provided).
  --solc SOLC_VERSION   Solidity compiler version (default: latest installed
                        version). Installed compiler versions are listed.
  --evm EVM_VERSION     Ethereum VM (default: 'petersburg'). Available VM's:
                        'homestead', 'byzantium' or 'petersburg'.
  --algorithm {confuzzius,divfuzz}
                        Optimization algorithm: 'confuzzius' (Adaptative
                        Genetic Algorithm, default) or 'divfuzz'
                        (divfuzz based).
  -g GENERATIONS, --generations GENERATIONS
                        Number of generations (default: 10).
  -t GLOBAL_TIMEOUT, --timeout GLOBAL_TIMEOUT
                        Number of seconds for fuzzer to stop.
  -n POPULATION_SIZE, --population-size POPULATION_SIZE
                        Size of the population.
  -pc PROBABILITY_CROSSOVER, --probability-crossover PROBABILITY_CROSSOVER
                        Probability of crossover (default: 0.8).
  -pm PROBABILITY_MUTATION, --probability-mutation PROBABILITY_MUTATION
                        Probability of mutation (default: 0.2).
  -dw DATA_DEPENDENCY_WEIGHT, --data-dependency-weight DATA_DEPENDENCY_WEIGHT
                        Weight of data dependency in the fitness score
                        (default: 1.0, used only with --algorithm divfuzz).
  -bw BRANCH_WEIGHT, --branch-weight BRANCH_WEIGHT
                        Weight of uncovered branches in the fitness score
                        (default: 1.0, used only with --algorithm divfuzz).
  -uw UNIQUE_BRANCH_WEIGHT, --unique-branch-weight UNIQUE_BRANCH_WEIGHT
                        Weight of unique branch coverage in the fitness score
                        (default: 1.0, used only with --algorithm divfuzz).
  -r RESULTS, --results RESULTS
                        Folder or JSON file where results should be stored.
  --tag TAG             Tag to prefix the result file name.
  --seed SEED           Initialize the random number generator with a given
                        seed.
  --cfg                 Build control-flow graph and highlight code coverage.
  --rpc-host RPC_HOST   Ethereum client RPC hostname.
  --rpc-port RPC_PORT   Ethereum client RPC port.
  --data-dependency DATA_DEPENDENCY
                        Enable/disable data dependency analysis:
                        0 - Disable, 1 - Enable (default: 0).
  --diversity DIVERSITY
                        Enable/disable diversity analysis:
                        0 - Disable, 1 - Enable (default: 0).
  --constraint-solving CONSTRAINT_SOLVING
                        Enable/disable constraint solving:
                        0 - Disable, 1 - Enable (default: 1).
  --environmental-instrumentation ENVIRONMENTAL_INSTRUMENTATION
                        Enable/disable environmental instrumentation:
                        0 - Disable, 1 - Enable (default: 1).
  --max-individual-length MAX_INDIVIDUAL_LENGTH
                        Maximal length of an individual (default: 5).
  --max-symbolic-execution MAX_SYMBOLIC_EXECUTION
                        Maximum number of symbolic execution calls before
                        resetting the population (default: 10).
  -v, --version         show program's version number and exit
```

#### Local Fuzzing (Off-Chain)

``` shell
python3 fuzzer/main.py -s examples/RemiCoin/contracts/RemiCoin.sol -c RemiCoin --solc v0.4.26 --evm byzantium -g 20 --algorithm confuzzius
```


#### Example with divfuzz Diversity Algorithm

``` shell
python3 fuzzer/main.py -s examples/TokenSale/contracts/TokenSale.sol -c TokenSale --solc v0.4.26 --evm byzantium --algorithm divfuzz -dw 1.2 -bw 0.8 -t 30
```
