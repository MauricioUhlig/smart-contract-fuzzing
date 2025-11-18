#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Crossover operator implementation. '''

import random

from utils import settings
from ...plugin_interfaces.operators.crossover import Crossover
from ...components.individual import Individual

class DiversityCrossover(Crossover):
    def __init__(self, pc):
        '''
        :param pc: The probability of crossover (usaully between 0.25 ~ 1.0)
        :type pc: float in (0.0, 1.0]
        '''
        if pc <= 0.0 or pc > 1.0:
            raise ValueError('Invalid crossover probability')

        self.pc = pc

    def cross(self, father, mother):
        """
        Enhanced crossover that promotes diversity with better performance
        """
        if random.random() > self.pc or mother is None:
            return father.clone(), mother.clone()
        
        _father = father.clone()
        _mother = mother.clone()
        
        max_length = settings.MAX_INDIVIDUAL_LENGTH
        
        # Early termination check
        if len(_father.chromosome) + len(_mother.chromosome) > max_length:
            return _father, _mother
        
        # Use slicing instead of deepcopy for chromosome access
        p1_genes = _father.chromosome[:]  # Shallow copy is sufficient if genes are immutable
        p2_genes = _mother.chromosome[:]
        
        # Pre-allocate lists with estimated capacity to avoid resizing
        estimated_len = min(len(p1_genes) + len(p2_genes), max_length)
        child1_chromosome = []  # Can pre-allocate: [None] * estimated_len
        child2_chromosome = []
        
        # Shuffle in place
        random.shuffle(p1_genes)
        random.shuffle(p2_genes)
        
        # Optimized alternating crossover
        min_len = min(len(p1_genes), len(p2_genes))
        
        # Process common length efficiently
        for i in range(min_len):
            child1_chromosome.append(p1_genes[i])
            child1_chromosome.append(p2_genes[i])
            child2_chromosome.append(p2_genes[i])
            child2_chromosome.append(p1_genes[i])
        
        # Handle remaining genes
        if len(p1_genes) > min_len:
            remaining = p1_genes[min_len:]
            child1_chromosome.extend(remaining)
            child2_chromosome.extend(remaining)
        elif len(p2_genes) > min_len:
            remaining = p2_genes[min_len:]
            child1_chromosome.extend(remaining)
            child2_chromosome.extend(remaining)
        
        # Trim to max length
        child1_chromosome = child1_chromosome[:max_length]
        child2_chromosome = child2_chromosome[:max_length]
        
        # Create children more efficiently
        child1 = Individual(generator=_father.generator)
        child1.init(chromosome=child1_chromosome)
        
        child2 = Individual(generator=_mother.generator)
        child2.init(chromosome=child2_chromosome)
        
        return child1, child2
