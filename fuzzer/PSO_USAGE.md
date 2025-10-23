# Particle Swarm Optimization (PSO) Usage Guide

## Overview

This fuzzer now supports **Particle Swarm Optimization (PSO)** as an alternative to the Genetic Algorithm (GA). PSO is a population-based optimization technique inspired by the social behavior of bird flocking or fish schooling.

## How PSO Works

In PSO, each individual (called a "particle") represents a potential solution. Particles move through the search space influenced by:

1. **Inertia (w)**: Tendency to continue in the current direction
2. **Cognitive Component (c1)**: Attraction toward the particle's personal best position
3. **Social Component (c2)**: Attraction toward the global best position found by any particle

### PSO vs Genetic Algorithm

| Aspect | Genetic Algorithm | Particle Swarm Optimization |
|--------|------------------|----------------------------|
| **Operators** | Selection, Crossover, Mutation | Velocity updates based on personal/global best |
| **Exploration** | Through crossover and mutation | Through inertia and social learning |
| **Memory** | No memory of past positions | Remembers personal best and global best |
| **Parameters** | Crossover rate, mutation rate | Inertia weight (w), cognitive (c1), social (c2) coefficients |

## Usage

### Basic PSO Command

```bash
python main.py --algorithm pso -s contract.sol -c ContractName
```

### PSO with Custom Parameters
python3 fuzzer/main.py -s examples/RemiCoin/contracts/RemiCoin.sol -c RemiCoin --solc v0.4.26 --evm byzantium -g 20 --algorithm pso
```bash 
python3 fuzzer/main.py \
  -s examples/RemiCoin/contracts/RemiCoin.sol \
  -c RemiCoin \
  --solc v0.4.26 \
  --evm byzantium \
  -g 20 \
  --algorithm pso
```
<!-- ```bash
python3 fuzzer/main.py --algorithm pso \
  -s examples/RemiCoin/contracts/RemiCoin.sol \
  -c RemiCoin \
  -g 50 \            # Number of generations
  -n 20 \            # Population size
  --solc v0.4.26 \
  --evm byzantium
  #--pso-w 0.7 \      # Inertia weight (default: 0.7)
  #--pso-c1 1.5 \     # Cognitive coefficient (default: 1.5)
  #--pso-c2 1.5 \     # Social coefficient (default: 1.5)
``` -->

### Using Genetic Algorithm (Default)

```bash
python main.py --algorithm ga -s contract.sol -c ContractName
```

Or simply omit the `--algorithm` flag:

```bash
python main.py -s contract.sol -c ContractName
```

## PSO Parameters

### Inertia Weight (w)
- **Range**: 0.0 - 1.0
- **Default**: 0.7
- **Effect**: Controls exploration vs exploitation
  - Higher values (0.8-1.0): More exploration, particles move more freely
  - Lower values (0.4-0.6): More exploitation, particles converge faster

### Cognitive Coefficient (c1)
- **Range**: 0.0 - 3.0
- **Default**: 1.5
- **Effect**: Controls attraction to personal best
  - Higher values: Particles trust their own experience more
  - Lower values: Particles rely less on personal history

### Social Coefficient (c2)
- **Range**: 0.0 - 3.0
- **Default**: 1.5
- **Effect**: Controls attraction to global best
  - Higher values: Particles follow the swarm more closely
  - Lower values: Particles are more independent

## Recommended Parameter Settings

### Balanced Exploration and Exploitation (Default)
```bash
--pso-w 0.7 --pso-c1 1.5 --pso-c2 1.5
```

### More Exploration (for complex contracts)
```bash
--pso-w 0.9 --pso-c1 2.0 --pso-c2 1.0
```

### More Exploitation (for quick convergence)
```bash
--pso-w 0.4 --pso-c1 1.0 --pso-c2 2.0
```

### Balanced with Social Emphasis
```bash
--pso-w 0.7 --pso-c1 1.2 --pso-c2 1.8
```

## Complete Example

```bash
python main.py \
  --algorithm pso \
  -s examples/SimpleDAO.sol \
  -c SimpleDAO \
  --pso-w 0.7 \
  --pso-c1 1.5 \
  --pso-c2 1.5 \
  -g 100 \
  -n 30 \
  --seed 42 \
  --cfg
```

## When to Use PSO vs GA

### Use PSO when:
- You want faster convergence to good solutions
- The search space has clear fitness gradients
- You want simpler parameter tuning (3 parameters vs multiple GA operators)
- Memory of good solutions is beneficial

### Use GA when:
- You need more diverse exploration
- The problem benefits from recombination of solutions
- You want to use data dependency analysis (currently GA-only)
- You need more control over genetic operators

## Implementation Details

The PSO implementation:
1. Initializes particles with random positions (transaction sequences)
2. Evaluates fitness for each particle
3. Tracks personal best for each particle and global best across all particles
4. Updates particle positions based on PSO velocity equations
5. Applies small perturbations for additional exploration

Each particle's "position" is a chromosome (sequence of transactions), and the PSO algorithm guides particles toward better transaction sequences that maximize code coverage.
