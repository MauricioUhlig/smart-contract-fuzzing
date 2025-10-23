# PySwarms Engine Usage Guide

## Overview

The PySwarms engine integrates the popular [PySwarms](https://pyswarms.readthedocs.io/) library into the smart contract fuzzer, providing access to well-tested and optimized Particle Swarm Optimization algorithms.

## Key Features

### 1. **Library-Based PSO**
- Uses the mature PySwarms library with proven optimization algorithms
- Access to multiple optimizer types and topologies
- Efficient numpy-based computations

### 2. **Optimizer Types**

#### Global Best PSO (`--pyswarms-optimizer global`)
- Each particle is influenced by the global best position found by the entire swarm
- Faster convergence but may get stuck in local optima
- Best for: Problems with clear global optimum

#### Local Best PSO (`--pyswarms-optimizer local`)
- Each particle is influenced by the best position in its local neighborhood
- Slower convergence but better exploration
- Configurable topology determines neighborhood structure
- Best for: Complex problems with multiple local optima

### 3. **Topology Options** (for Local Best PSO)

- **Star** (`--pyswarms-topology star`): All particles connected to a central hub
- **Ring** (`--pyswarms-topology ring`): Particles connected in a circular chain
- **Pyramid** (`--pyswarms-topology pyramid`): Hierarchical pyramid structure
- **Random** (`--pyswarms-topology random`): Random connections between particles

### 4. **Continuous-to-Discrete Encoding**
- Automatically encodes discrete chromosome representations into continuous space
- Maps PSO positions back to valid transaction sequences
- Handles variable-length chromosomes and complex gene structures

## Usage Examples

### Basic PySwarms (Global Best)

```bash
python main.py \
  --source contract.sol \
  --contract MyContract \
  --algorithm pyswarms \
  --generations 50
```

### Local Best PSO with Ring Topology

```bash
python main.py \
  --source contract.sol \
  --contract MyContract \
  --algorithm pyswarms \
  --pyswarms-optimizer local \
  --pyswarms-topology ring \
  --generations 50
```

### Custom Hyperparameters

```bash
python main.py \
  --source contract.sol \
  --contract MyContract \
  --algorithm pyswarms \
  --pyswarms-w 0.9 \
  --pyswarms-c1 2.0 \
  --pyswarms-c2 2.0 \
  --generations 100
```

### Comparing Optimizers

```bash
# Global Best PSO
python main.py -s contract.sol -c MyContract --algorithm pyswarms --pyswarms-optimizer global -g 50

# Local Best PSO with Star topology
python main.py -s contract.sol -c MyContract --algorithm pyswarms --pyswarms-optimizer local --pyswarms-topology star -g 50

# Local Best PSO with Ring topology
python main.py -s contract.sol -c MyContract --algorithm pyswarms --pyswarms-optimizer local --pyswarms-topology ring -g 50
```

## Parameters

### PySwarms-Specific Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--pyswarms-optimizer` | Optimizer type (global/local) | `global` | global, local |
| `--pyswarms-topology` | Topology for local PSO | `star` | star, ring, pyramid, random |
| `--pyswarms-w` | Inertia weight | `0.7` | 0.4 - 0.9 |
| `--pyswarms-c1` | Cognitive coefficient | `1.5` | 1.0 - 3.0 |
| `--pyswarms-c2` | Social coefficient | `1.5` | 1.0 - 3.0 |

### General Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-g, --generations` | Number of iterations | 10 |
| `-n, --population-size` | Number of particles | 2 × functions |
| `-t, --timeout` | Time limit in seconds | None |

## How It Works

### 1. Encoding Phase
- Each individual's chromosome (transaction sequence) is encoded into a continuous position vector
- Dimensions include: account indices, amounts, gas limits, function selectors, and arguments
- All values normalized to [0, 1] range

### 2. Optimization Phase
- PySwarms optimizer explores the continuous search space
- Each position is decoded back to a valid individual
- Fitness function evaluates code coverage
- Optimizer updates positions based on personal and global/local best

### 3. Decoding Phase
- Final positions are decoded back to transaction sequences
- Population is updated with optimized individuals

## When to Use PySwarms

### Advantages
- **Mature library**: Well-tested, optimized implementation
- **Multiple topologies**: Experiment with different neighborhood structures
- **Efficient**: Numpy-based vectorized operations
- **Flexible**: Easy to tune hyperparameters

### Best For
- Benchmarking against standard PSO implementations
- Experimenting with different PSO variants
- Problems where continuous optimization works well
- When you need proven, reliable PSO algorithms

### Limitations
- Encoding/decoding overhead for discrete problems
- May not leverage domain-specific knowledge as well as custom operators
- Fixed optimization strategy (can't easily customize update rules)

## Comparison with Other Algorithms

| Algorithm | Convergence | Exploration | Diversity | Best For |
|-----------|-------------|-------------|-----------|----------|
| **GA** | Medium | Good | Medium | General purpose, well-understood |
| **Custom PSO** | Fast | Medium | Low | Quick convergence, simple problems |
| **PySwarms** | Fast | Good | Medium | Standard PSO, benchmarking |
| **Collaborative** | Slow | Excellent | High | Complex contracts, maximum coverage |

## Tips for Best Results

1. **Start with Global Best**: Try global optimizer first for faster results
2. **Use Local Best for complex contracts**: Switch to local with ring/pyramid topology for better exploration
3. **Tune inertia weight**: Higher w (0.8-0.9) for more exploration, lower (0.4-0.6) for exploitation
4. **Balance c1 and c2**: Equal values (1.5-2.0) work well for most problems
5. **Increase population size**: More particles = better exploration but slower iterations
6. **Run longer**: PySwarms may need more iterations to converge than GA

## Installation

PySwarms is automatically installed via requirements.txt:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pyswarms==1.3.0
```

## References

- [PySwarms Documentation](https://pyswarms.readthedocs.io/)
- [PySwarms GitHub](https://github.com/ljvmiranda921/pyswarms)
- Original PSO paper: Kennedy & Eberhart (1995)
