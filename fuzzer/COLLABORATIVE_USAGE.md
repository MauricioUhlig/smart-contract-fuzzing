# Collaborative Diversity Engine Usage Guide

## Overview

The Collaborative Diversity Engine is a novel evolutionary algorithm that focuses on maintaining population diversity to achieve better group-fitness scores. Unlike traditional genetic algorithms that optimize individual fitness, this approach considers the collective performance of the entire population.

## Key Features

### 1. Population-Based Fitness Function
- **Individual Fitness**: Traditional branch coverage fitness
- **Unique Contribution Bonus**: Rewards individuals covering branches not covered by others
- **Complementarity Bonus**: Rewards individuals covering rarely-covered branches
- **Formula**: `Total Fitness = Individual Fitness + (Unique + Complementarity) × Diversity Weight`

### 2. Diversity-Based Selection
- Uses tournament selection with diversity consideration
- Balances fitness and behavioral diversity
- Prevents premature convergence to local optima
- Maintains population variety throughout evolution

### 3. Enhanced Crossover for Diversity
- Interleaved crossover that maximizes offspring diversity
- Shuffles parent genes before combining
- Alternates genes from both parents
- Creates children that explore different solution spaces

### 4. Diversity-Aware Mutation
- Adaptive mutation rates based on gene frequency
- Higher mutation for common genes (promotes exploration)
- Lower mutation for rare genes (preserves diversity)
- Dynamic add/remove operations for chromosome length variation

### 5. Archive of Diverse Solutions
- Maintains archive of novel high-performing solutions
- Novelty threshold prevents similar solutions
- Preserves diverse approaches to coverage
- Maximum archive size: 20 solutions

## Usage

### Basic Command

```bash
python main.py \
  --source contract.sol \
  --contract ContractName \
  --algorithm collaborative \
  -g 50 \
  -n 20
```

### Advanced Configuration
```bash
python3 fuzzer/main.py \
  --source examples/RemiCoin/contracts/RemiCoin.sol \
  --contract RemiCoin \
  --algorithm collaborative \
  --generations 100 \
  --population-size 30 \
  --diversity-weight 0.4 \
  --novelty-threshold 0.6 \
  --probability-crossover 0.9 \
  --probability-mutation 0.15 \
  --solc v0.4.26 \
  --evm byzantium
```

## Parameters

### Algorithm Selection
- `--algorithm collaborative` - Use Collaborative Diversity Engine

### Diversity Parameters
- `--diversity-weight <float>` - Weight for diversity in fitness calculation (default: 0.3)
  - Range: 0.0 to 1.0
  - Higher values prioritize diversity over individual fitness
  - Recommended: 0.2 - 0.5

- `--novelty-threshold <float>` - Behavioral distance threshold for archive novelty (default: 0.5)
  - Range: 0.0 to 1.0
  - Higher values require more diverse solutions for archive
  - Recommended: 0.4 - 0.7

### Standard Evolutionary Parameters
- `-g, --generations <int>` - Number of evolution iterations (default: 10)
- `-n, --population-size <int>` - Population size (must be even)
- `-pc, --probability-crossover <float>` - Crossover probability (default: 0.9)
- `-pm, --probability-mutation <float>` - Base mutation probability (default: 0.1)

## How It Works

### Evolution Loop

1. **Execute Population**: Run all individuals and collect coverage data
2. **Calculate Collaborative Fitness**: 
   - Compute individual fitness (branch coverage)
   - Add unique contribution bonus
   - Add complementarity bonus
3. **Measure Diversity**: Calculate population-wide behavioral diversity
4. **Update Archive**: Store novel high-performing solutions
5. **Create New Generation**:
   - Select parents using diversity-based tournament
   - Apply diversity-promoting crossover
   - Apply adaptive mutation
6. **Repeat** until generations complete or timeout

### Diversity Metrics

**Behavioral Distance**: Measures how different two individuals are based on:
- Chromosome length differences
- Function call differences
- Account usage differences

**Population Diversity**: Average behavioral distance between all pairs of individuals

### Fitness Components

**Individual Fitness**:
- Branch coverage score (standard)
- Data dependency coverage (if enabled)

**Group Contribution**:
- Unique branches: Branches only this individual covers
- Complementarity: Branches covered by <30% of population

## When to Use Collaborative Engine

### Best For:
- Complex contracts with many execution paths
- Contracts where diverse test cases are needed
- Avoiding premature convergence
- Maximizing overall coverage across population
- Finding rare edge cases

### Compared to GA:
- **GA**: Fast convergence, may miss rare paths
- **Collaborative**: Slower but more thorough, better diversity

### Compared to PSO:
- **PSO**: Good for continuous optimization
- **Collaborative**: Better for discrete test case generation

## Example Scenarios

### Scenario 1: High Diversity Needed
```bash
# Maximize diversity for complex contract
python main.py \
  --source complex_contract.sol \
  --contract ComplexContract \
  --algorithm collaborative \
  --diversity-weight 0.5 \
  --novelty-threshold 0.7 \
  -g 100 \
  -n 40
```

### Scenario 2: Balanced Approach
```bash
# Balance fitness and diversity
python main.py \
  --source contract.sol \
  --contract MyContract \
  --algorithm collaborative \
  --diversity-weight 0.3 \
  --novelty-threshold 0.5 \
  -g 50 \
  -n 20
```

### Scenario 3: Quick Exploration
```bash
# Fast diverse exploration
python main.py \
  --source contract.sol \
  --contract MyContract \
  --algorithm collaborative \
  --diversity-weight 0.2 \
  -g 30 \
  -n 16 \
  -pm 0.2
```

## Output Interpretation

### Log Messages

```
Generation 1/50 | Best Fitness: 15.20 | Avg Fitness: 8.45 | Diversity: 0.623 | Coverage: 42 | Time: 2.31s
```

- **Best Fitness**: Highest collaborative fitness in population
- **Avg Fitness**: Average collaborative fitness
- **Diversity**: Population diversity score (0-1, higher is more diverse)
- **Coverage**: Total unique branches covered
- **Time**: Generation execution time

### Final Summary

```
Evolution completed in 125.43s
Final coverage: 87 branches
Archive size: 18 diverse solutions
```

- **Final coverage**: Total branches discovered
- **Archive size**: Number of diverse solutions preserved

## Tips for Best Results

1. **Start with default parameters** and adjust based on results
2. **Increase diversity_weight** if population converges too quickly
3. **Increase novelty_threshold** to maintain more diverse archive
4. **Larger populations** (30-50) work better for complex contracts
5. **Higher mutation rates** (0.15-0.2) promote more exploration
6. **Monitor diversity metric** - should stay above 0.3 for good exploration

## Troubleshooting

**Low diversity (<0.2)**:
- Increase `--diversity-weight`
- Increase `--probability-mutation`
- Increase population size

**Slow convergence**:
- Decrease `--diversity-weight`
- Increase population size
- Increase generations

**Low coverage**:
- Increase generations
- Adjust `--diversity-weight` (try both higher and lower)
- Enable data dependency analysis with `--data-dependency 1`
