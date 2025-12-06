#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Mutation implementation. '''

import random

from utils import settings
from ...plugin_interfaces.operators.mutation import Mutation
from engine.fitness import fitness_function
from engine.environment import FuzzingEnvironment
import copy

class DiversityMutation(Mutation):
    def __init__(self, pm):
        '''
        :param pm: The probability of mutation (usually between 0.001 ~ 0.1)
        :type pm: float in (0.0, 1.0]
        '''
        if pm <= 0.0 or pm > 1.0:
            raise ValueError('Invalid mutation probability')

        self.pm = pm

    def mutate(self, individual, engine):
        """
        Diversity-aware mutation that introduces novel variations
        Higher mutation rate for genes that are common in population
        """
        current_fitness = engine.current_fitness_base_scores[individual.hash]
         # Calculate gene frequency in population
        gene_frequencies = {}
        for ind in engine.population.individuals:
            for gene in ind.chromosome:
                func = gene["arguments"][0]
                gene_frequencies[func] = gene_frequencies.get(func, 0) + 1
        
        # Mutate each gene with adaptive probability
        for i, gene in enumerate(individual.chromosome):
            func = gene["arguments"][0]
            frequency = gene_frequencies.get(func, 0) / len(engine.population.individuals)
            
            # Higher mutation rate for common genes
            adaptive_pm = engine.args.probability_mutation * (1 + frequency)
            
            if random.random() < adaptive_pm:
                # Mutate this gene
                mutation_type = random.choice([
                    "account", "amount", "arguments", "timestamp", 
                    "blocknumber", "gaslimit"
                ])
                
                if mutation_type == "account":
                    gene["account"] = engine.generator.get_random_account(func)
                elif mutation_type == "amount":
                    gene["amount"] = engine.generator.get_random_amount(func)
                elif mutation_type == "arguments":
                    if len(gene["arguments"]) > 1:
                        arg_idx = random.randint(1, len(gene["arguments"]) - 1)
                        # if arg_idx < len(gene["arguments"]):
                        arg_type = engine.env.interface[func][arg_idx - 1]
                        gene["arguments"][arg_idx] = engine.generator.get_random_argument(
                            arg_type, func, arg_idx - 1
                        )
                elif mutation_type == "timestamp":
                    gene["timestamp"] = engine.generator.get_random_timestamp(func)
                elif mutation_type == "blocknumber":
                    gene["blocknumber"] = engine.generator.get_random_blocknumber(func)
                elif mutation_type == "gaslimit":
                    gene["gaslimit"] = engine.generator.get_random_gaslimit(func)
        
        # # Add/remove genes to promote diversity
        # if random.random() < engine.args.probability_mutation:
        #     if len(individual.chromosome) < settings.MAX_INDIVIDUAL_LENGTH and random.random() < 0.5:
        #         # Add a new gene
        #         new_gene = engine.generator.generate_random_input()
        #         individual.chromosome.append(new_gene)
        #     elif len(individual.chromosome) > 1 and random.random() < 0.5:
        #         # Remove a gene
        #         individual.chromosome.pop(random.randint(0, len(individual.chromosome) - 1))
        
        # Recalculate hash
        new_individual = individual.clone()
        new_individual.init(individual.chromosome)

        # new_env = copy.deepcopy(engine)
        engine.analysis[0].execution_function(new_individual, engine.env)
        new_fitness = fitness_function(new_individual, engine.env)
        if new_fitness > current_fitness:
            return new_individual
        else:
            return individual
