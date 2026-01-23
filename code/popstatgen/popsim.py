'''
This file contains classes related to population and statistical genetics.
The classes present are:
- Population: Models a population of individuals with genotypes. Attributes store raw and summary data on such individuals, including traits.
- Trait: Models a trait in a population, including its individual components.
These classes largely contain wrapper methods that call functions from other files. These wrapper methods extract the necessary attributes from the class object and pass them as arguments to the functions.
These wrapper methods often just update the class attributes, instead of returning the results.
'''

# imports ####
# package imports
from . import core_functions as core
from . import popgen_functions as pop
from . import statgen_functions as stat
# other imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Dict, Optional, List
from dataclasses import dataclass
from scipy import sparse
from scipy.linalg import block_diag
import copy
import inspect
import warnings

class Population:
    '''
    Class for a population to simulate. Contains genotype information. Contains methods to simulate change in population over time.
    '''

    ########################
    #### Initialization ####
    ########################

    def __init__(self, N: int, M: int, P: int = 2,
                 p_init: Union[float, np.ndarray] = None,
                 keep_past_generations: int = 1,
                 track_pedigree: bool = False,
                 seed: int = None):
        '''
        Initializes a population, simulating initial genotypes from specified allele frequencies.
        Parameters:
            N (int): Population size of individuals (not haplotypes).
            M (int): Total number of variants in genome.
            P (int): Ploidy of genotpes. Default is 2 (diploid).
            p_init (float or array): Initial allele frequency of variants. If only a single value is provided, it is treated as the initial allele frequency for all variants. Alternatively, can be an array of length M for variant-specfic allele frequencies. If not provided, default is uniform distribution of allele frequencies between 0.05 and 0.95.
            keep_past_generations (int): Number of past generations to keep in the object. Default is 1, meaning the past generation is kept (on top of the current generation).
            track_pedigree (bool): Whether to track pedigree information (stored in Population.ped). Must keep at least 1 past generation. Default is False.
            seed (int): Initial seed to use when simulating genotypes (and allele frequencies if necessary).
        '''
        # sets seed if specified
        if seed is not None:
            seed = np.random.seed(seed)
        
        # draws initial allele frequencies from uniform distribution between 0.05 and 0.95 if not specified
        if p_init is None:
            p_init = pop.draw_p_init(M, method = 'uniform', params = (0.05, 0.95))
        elif type(p_init) == float or type(p_init) == int:
            # if only single value is given, all variants have same initial frequency
            p_init = np.full(M, p_init)

        # generates initial genotypes and records allele frequencies
        H = pop.draw_binom_haplos(p_init, N, P)

        # passes haplotype to base constructor for further initialization
        self._initialize_H(H, keep_past_generations=keep_past_generations, track_pedigree=track_pedigree)
    
    @classmethod
    def from_H(cls, H: np.ndarray, keep_past_generations: int = 1, track_pedigree: bool = False):
        '''
        Initializes a population from a given haplotype array.
        Parameters:
            H (3D array): N*M*P array of haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
            keep_past_generations (int): Number of past generations to keep in the object. Default is 1, meaning the past generation is kept (on top of the current generation).
            track_pedigree (bool): Whether to track pedigree information (stored in Population.ped). Must keep at least 1 past generation. Default is False.
        Returns:
            Population: A new Population object initialized with the given haplotype array.
        '''
        # creates new instance of class
        pop = cls.__new__(cls)
        # initializes the object with the given haplotype array
        pop._initialize_H(H, keep_past_generations=keep_past_generations, track_pedigree=track_pedigree)
        return pop

    def _initialize_H(self, H: np.ndarray, keep_past_generations: int = 1, track_pedigree: bool = False):
        '''
        Initializes a population from a given haplotype array. See the `from_H` class method for details.
        '''
        # sets basic population attributes from haplotype array
        (self.N, self.M, self.P) = H.shape

        # initializes default/initial attributes
        self.t = 0 # generation
        self.T_breaks = [self.t] # simulation breaks
        self.traits = {}
        self.BPs = np.arange(self.M) # variant positions in base pairs (BPs)
        self.R = pop.generate_LD_blocks(self.M) # recombination rates
        self.K = np.diag(np.ones(self.N)) # kinship matrix (initially identity, not functional yet)
        self.track_pedigree = track_pedigree # whether to track pedigree information
        # how many past generations to keep in memory
        self.past = [self]
        self.update_keep_past_gens(keep_past_generations=keep_past_generations)
        Haplos = np.full_like(H, -1, dtype=int) # initializes haplotype ID array with -1s

        # further attributes
        self._update_obj(H=H, Haplos=Haplos, update_past=False)
        self.assign_sex() # assigns sex (F:0 / M:1)
        self.relations = pop.initialize_relations(self.N)
        self.ped = Pedigree(self.N)

        # defines metrics
        self.metric = {}
        self._define_metric('p', pop.compute_freqs, shape = [self.M], # allele frequency
                            P = self.P)
        self._initialize_metrics(G = self.G)
    
    def update_keep_past_gens(self, keep_past_generations: int):
        '''
        Small function for structuring the 'past' attribute to reflect the number of generations in the past that are stored in the current population object. If the number of past generations is reduced, the older generations are removed from the array. If the number if past generations is increased, 'past' array is padded with Nones for future compatibility.
        Parameters:
            keep_past_generations (int): Number of past generations to keep in the object.
        '''
        current_past_gens = len(self.past) - 1 # should also just be self.keep_past_generations
        if keep_past_generations < current_past_gens:
            self.past = self.past[0:keep_past_generations + 1]
        elif keep_past_generations > current_past_gens:
            for _ in range(current_past_gens, keep_past_generations):
                self.past.append(None) # initializes past generations' objects as None
        elif keep_past_generations > 0:
            warnings.warn(f'Number of past generations kept is already {keep_past_generations}')
        
        self.keep_past_generations = keep_past_generations


    def make_sites_indep(self):
        '''
        Changes the recombination rates to make all sites independent.
        '''
        self.R = 0.5 * np.ones(self.M)

    def set_founding_haplotypes(self):
        '''
        Generates a complementary haplotype array for each individual containing a haplotype identifier for each allele. This functions treats the current generation as founders (individuals are unrelated from each other) such that each of their chromosomes has a unique identifier for all alleles in it. Subsequent generations can then track the inheritance of these founding haplotypes. Haplotypes are given an integer in the order they appear in the haplotype array.
        '''
        ids = np.arange(self.N * self.P, dtype=np.int32).reshape(self.N, self.P)
        Haplos = np.broadcast_to(ids[:, None, :], self.H.shape).copy()
        self._update_obj(Haplos=Haplos)


    ###################################
    #### Storing object attributes ####
    ###################################

    def _update_obj(self, H: np.ndarray = None, update_past: bool = True,
                    relations: dict = None, Haplos: np.ndarray = None,
                    update_pedigree: bool = False):
        '''
        Update the population object's attributes.
        Parameters:
            H (3D array): Haplotype array.
            update_past (bool): Whether to update the memory of past generations. Default is True.
            relations (dict): Dictionary of relationship matrices to update. If None, does not update relationships. Default is None.
            Haplos (3D array): Haplotype identifier array. If None, does not update haplotype identifiers. Default is None.
            update_pedigree (bool): Whether to update pedigree information. Must also update past generations. Default is False.
        '''
        # keep this before updating anything else
        if update_past:
            self._update_past()
        
        if H is not None:
            self.H = H
            self.G = pop.make_G(self.H)
            self.p = pop.compute_freqs(self.G, self.P)
            self.X = pop.standardize_G(self.G, self.p, self.P, impute=True, std_method='observed')
        if Haplos is not None:
            self.Haplos = Haplos
        if relations is not None:
            self._update_relations(relations)
        if update_pedigree and update_past:
            # this must happen after updating past generations and their relations
            self._update_pedigree()

    def _update_past(self):
        '''
        Updates the past generations' objects, shifting them by one and adding the current object as the first element.
        '''
        if self.keep_past_generations > 0:
            # makes copy of current object
            gen_t1 = copy.deepcopy(self)
            # removes unnecessary attributes (e.g. past generations)
            gen_t1.past = None
            # shifts previous generations
            self.past[2:] = self.past[1:self.keep_past_generations]
            self.past[1] = gen_t1
            self.past[0] = self
        self.t += 1
    
    def _update_relations(self, relations: dict):
        '''
        Updates the relationships in the population object.
        Parameters:
            relations (dict): Dictionary of relationship matrices to update. Keys are relationship types (e.g. 'spouses', 'parents') and values are the corresponding matrices.
        '''
        for key in relations:
            if key == 'spouses':
                if self.keep_past_generations >= 1:
                    self.past[1].relations['spouses'] = relations['spouses'] # updates prior spouse matrix for past gen
                self.relations['spouses'] = np.zeros((self.N, self.N), dtype=np.uint8) # resets spouses relationship matrix for current gen
            else:
                self.relations[key] = relations[key] # sets other relationship matrices for current generation

    def _update_pedigree(self):
        '''
        Updates the pedigree information in the population object for a new generation.
        '''
        if self.keep_past_generations < 1:
            raise Exception('Must keep at least 1 past generation to track pedigree.')
        # makes new Pedigree object from scratch, and adds parent infices and Pedigree pointer
        self.ped = Pedigree(self.N, par_idx = self.relations['par_idx'], par_Ped = self.past[1].ped)
        #self.ped.construct_rel4(self.relations['par_idx'], self.past[1].ped)
        self.ped.construct_rel2()
        #self.ped.fill_rel2_from_rel4()

    def store_neighbor_matrix(self, LDwindow: float = None) -> sparse.coo_matrix:
        '''
        Stores boolean sparse matrix of variants (`neighbor_matrix`) within the specified LD window distance (`LDwindow`) based on object's variant positions (currently only supports `BPs`). Calls `make_neighbor_matrix()` from `popgen_functions.py`.
        Parameters:
            LDwindow (float): Maximum distance between variants to be considered neighbors. In the same units as the positions used. If not provided, defaults to infinite maximum distance.
        '''
        # by default, uses object's base pair positions
        positions = self.BPs
        self.neighbor_matrix = pop.make_neighbor_matrix(positions=positions, LDwindow=LDwindow)

    def store_LD_matrix(self):
        '''
        Stores the correlation matrix (`corr_matrix`) and its square, the LD matrix (`LD_matrix`). Must have pre-computed `neighbor_matrix` attribute (see `store_neighbor_matrix()`).

        Parameters:
            corr_matrix (sparse 2D matrix (CSR)): M*M correlation matrix between pairs of variants in CSR format. Can be computed with `get_corr_matrix()`.
        Returns:
            LD_matrix (sparse 2D matrix (CSR)): M*M LD matrix (square of correlation) between pairs of variants in CSR format.
        '''
        if not hasattr(self, 'neighbor_matrix'):
            raise Exception('Must have pre-computed neighbor_matrix. Use `Population.store_neighbor_matrix()`.')
        self.corr_matrix = pop.compute_corr_matrix(self.X, self.neighbor_matrix)
        self.LD_matrix = pop.compute_LD_matrix(self.corr_matrix)

    def store_GRM(self):
        '''
        Stores the genetic relationship matrix (GRM) in the `GRM` attribute using object's standardized genotype matrix. Calls `compute_GRM()` from `popgen_functions.py`.
        '''
        self.GRM = pop.compute_GRM(self.X)

    def add_trait(self, name: str, seed: int = None, **kwargs):
        '''
        Initializes and generates trait.
        Parameters:
            name (str): Name of trait.
            seed (int): Seed for random number generation.
            **kwargs: All other arguments are passed to the Trait constructor. See Trait.__init__ for details.
        '''
        if seed is not None:
            np.random.seed(seed)
        self.traits[name] = Trait(self.G, **kwargs)
        # stores actual heritability value
        self.traits[name].h2_true = self.traits[name].get_h2_true()
        self.traits[name].h2_trues = np.expand_dims(self.traits[name].h2_true, axis=0)
    
    def add_trait_from_effects(self, name: str, **kwargs):
        '''
        Initializes and generates trait from specified effects.
        Parameters:
            name (str): Name of trait.
            **kwargs: All other arguments are passed to the Trait.from_effects() constructor.
        '''
        self.traits[name] = Trait.from_effects(self.G, **kwargs)
        # stores actual heritability value
        self.traits[name].h2_true = self.traits[name].get_h2_true()
        self.traits[name].h2_trues = np.expand_dims(self.traits[name].h2_true, axis=0)
    
    def add_trait_from_fixed_values(self, name: str, y: np.ndarray):
        '''
        Initializes and generates trait from specified fixed trait values. No sub component or effects (e.g. genetic, random effects) are stored.
        Parameters:
            name (str): Name of trait.
            y (1D array): Trait values for each individual in the population.
        '''
        self.traits[name] = Trait.from_fixed_values(y)

    def update_traits(self, fixed_h2: bool = True, traits: list = None):
        '''
        Updates all traits by generating based on the current genotype matrix. Random noise components are re-generated. Causal genetic effects remain fixed. Only updates traits of composite type, except for sex, which is assigned through assign_sex().

        Parameters:
            fixed_h2 (bool): Whether the variance of the noise component should be updated to maintain the heritability. Default is True.
            traits (list of str): List of trait names to update. If None, updates all traits in the object.
        '''
        if traits is None:
            traits = self.traits.keys()
        # loops through each trait
        for key in self.traits:
            if key not in traits:
                continue
            trait = self.traits[key]
            if trait.type == 'composite':
                # trait object already contains allelic effects, which are applied to new genotypes
                trait.generate_trait(self.G, fixed_h2=fixed_h2)
            elif key == 'sex':
                self.assign_sex()

    def assign_sex(self):
        '''
        Randomly assigns half of individuals to being female (0) and other half to being male (1). Requires population size to be even.
        '''
        if self.N % 2 != 0:
            raise Exception('Population size must be even to assign sex equally and for mating purposes.')
        sex_arr = np.zeros(self.N, dtype=np.uint8)
        # randomly assigns exactly half of indices to be 1s
        sex_arr[np.random.choice(self.N, self.N // 2, replace=False)] = 1
        # adds sex as a Trait object (with name 'sex')
        self.add_trait_from_fixed_values(name='sex', y=sex_arr)

    #######################################
    #### Analysis of object attributes ####
    ####################################### 
    
    def _define_metric(self, metric_name: str, metric_func: callable, shape: list, **kwargs):
        '''
        Defines a metric for the population object. The metric is computed using the provided function and stored in the `metric` attribute of the object.

        Parameters:
            metric_name (str): Name of the metric to define.
            metric_func (callable): Function that computes the metric.
            shape (list): A list specifying the shape of the metric output at each generation. So for example, a metric like allele frequency would have [M] since at each generation, there is an allele frequency for each variant. A metric like population size would have [1] since it is a single value at each generation. Can be greater than length 1.
            **kwargs: Additional keyword arguments to pass to the metric function. These should be fixed settings of the metric function, not parameters that change over time.
        '''
        self.metric[metric_name] = {
            'func': metric_func, # the base function that computes the metric
            'active': True, # whether the metric is run per generation
            'shape': shape, # the shape of the metric output at each generation
            'valid_keys': set(inspect.signature(metric_func).parameters.keys()), # the valid keys for the metric function
            'kwargs': kwargs, # the fixed settings of the metric function
            'values': None # the values of the metric over generations, initialized to None
            }
    
    def _prep_metrics(self, generations: int):
        '''
        For each active metric, makes temporary array to store metric values over generations of simulation.
        Parameters:
            generations (int): Number of new generations to prepare each metric for.
        '''
        for metric_name in self.metric:
            if self.metric[metric_name]['active']:
                # adds number of generations to first dimension of metric history shape
                shape = [generations] + self.metric[metric_name]['shape']
                # initializes values to NaN
                self.metric[metric_name]['temp'] = np.full(shape, np.nan)

    def _update_temp_metrics(self, t: int, **kwargs):
        '''
        Updates the temporary metrics for the current generation. This is done by calling the metric function with the current generation's data and storing the result in the temporary metric array. The method `_prep_metrics` should be called before this method to prepare the temporary metric arrays.
        Parameters:
            t (int): Generation index for new batch of simulated generations. That is, instead of the population's current generation, it should be the index of the generation within the pre-specified number of new generations to simulate by `simulate_generations()`.
            **kwargs: All extra arguments that are passed to any metric function. Only the parameters needed for each metric function are passed to that function. The pre-specified arguments set in `_define_metric()` are automatically passed to each metric function.
        '''
        for metric_name in self.metric:
            if self.metric[metric_name]['active']:
                fixed_args = self.metric[metric_name]['kwargs']
                new_args = {k: v for k, v in kwargs.items() if k in self.metric[metric_name]['valid_keys']}
                args = {**new_args, **fixed_args}
                # runs metric
                metric_output = self.metric[metric_name]['func'](**args)
                # forces single-value metrics to be in array format
                if not isinstance(metric_output, np.ndarray):
                    metric_output = np.array([metric_output])
                # updates metric values in temporary array
                self.metric[metric_name]['temp'][t,] = metric_output
        
    def _update_metric_history(self):
        '''
        Updates the metric history for each active metric by appending the temporary metric values to the permanent metric values stored in the object.
        '''
        for metric_name in self.metric:
            if self.metric[metric_name]['active']:
                # gets temporary metric values
                temp_values = self.metric[metric_name]['temp']
                # appends to permanent values
                if self.metric[metric_name]['values'] is None:
                    self.metric[metric_name]['values'] = temp_values
                else:
                    self.metric[metric_name]['values'] = np.concatenate((self.metric[metric_name]['values'], temp_values), axis=0)
                # clears temporary metric values
                self.metric[metric_name].pop('temp', None)

    def _initialize_metrics(self, **kwargs):
        '''
        Initializes the metric history for each active metric by computing metrics for the current (starting) generation and storing them in the metric values. Should be called after defining metrics with `_define_metric()`.
        Parameters:
            **kwargs: All extra arguments that are passed to any metric function. Only the parameters needed for each metric function are passed to that function. The pre-specified arguments set in `_define_metric()` are automatically passed to each metric function.
        '''
        self._prep_metrics(1)
        self._update_temp_metrics(0, **kwargs)
        self._update_metric_history()
    
    ####################################
    #### Simulating forward in time ####
    ####################################
        
    def next_generation(self, s: Union[float, np.ndarray] = 0.0,
                        mu: Union[float, np.ndarray] = 0.0) -> np.ndarray:
        '''
        Simulates new generation. Doesn't simulate offspring directly, meaning that future offspring have haplotypes drawn randomly from allele frequencies. Automatically updates object.

        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float or 1D array): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
        Returns:
            H (3D array): N*M*P array of next generation's haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
        '''
        # assigns same selection coefficient/mutation rate to all variants if only single value specified
        if isinstance(s, (float, int)):
            s = np.full(self.M, s)
        if isinstance(mu, (float, int)):
            mu = np.full(self.M, mu)
        # analytical allele frequency from which next generations' alleles are drawn from
        p = self.p
        # effect of selection
        p = p * (1 + s) / (1 + p*s)
        # effect of mutation (in gametes that lead to new generation, i.e. post-selection)
        p = p*(1-mu) + (1-p)*mu
        # effect of genetic drift
        H = pop.draw_binom_haplos(p, self.N, self.P)

        return H

    def simulate_generations(self, generations: int, related_offspring: bool = False,
                             trait_updates: bool = False, fixed_h2: bool = False,
                             **kwargs):
        '''
        Simulates specified number of generations beyond current generation. Can simulate offspring directly. Automatically updates object. Recombination rates are extracted from object attributes.

        Parameters:
            generations (int): Number of generations to simulate (beyond the current generation).
            related_offspring (bool): Whether the offspring of the next generation should be directly related to parents from previous generation by simulating meiosis and haplotype transfer. Default is False, meaning that future offspring have haplotypes drawn randomly from allele frequencies.
            trait_updates (bool): Whether to update traits after each generation. However, sex is always updated in each generation, regardless of this setting. Default is False, meaning that traits are only updated at the end of the simulation.
            fixed_h2 (bool): Whether the variance of the noise component should be updated to maintain the heritability when updating traits. Default is False.
            **kwargs: All other arguments are passed to the `next_generation` or `generate_offspring` methods. See those methods for details.
        '''
        # preps metrics for new generations
        self._prep_metrics(generations)
        previous_gens = self.metric['p']['values'].shape[0]
        
        # loops through each generation
        for t in range(generations):
            if related_offspring:
                (H, relations, Haplos) = self.generate_offspring(**kwargs)
            else:
                H = self.next_generation(**kwargs)
                Haplos = np.full_like(H, -1, dtype=int) # unrelated haplotypes
                relations = None
            # updates objects and past
            self._update_obj(H=H, update_past=True, relations=relations, Haplos=Haplos, update_pedigree=self.track_pedigree)
            # records metrics
            self._update_temp_metrics(t, G=self.G)
            if trait_updates:
                self.update_traits(fixed_h2=fixed_h2)
            else:
                self.update_traits(traits=['sex']) # always updated in each generation
        self.T_breaks.append(previous_gens + generations)
        if not trait_updates:
            self.update_traits(fixed_h2=fixed_h2)
        # saves metrics to object
        self._update_metric_history()

    def _pair_mates(self, AM_r: float = 0, AM_trait: Union[str, np.ndarray] = None,
                    AM_type: str = 'phenotypic') -> np.ndarray:
        '''
        Pairs individuals up monogamously to mate and generate offspring. Population size must be multiple of 2. Allows for assortative mating (AM) if specified. Also updates the spouses relationship matrix.
        Parameters:
            AM_r (float): Desired correlation between AM trait values of spouses. Default is 0 (no assortative mating).
            AM_trait (str or 1D array): If a string, the name of the trait to use for assortative mating (as stored in the object). If an array, the trait values to use for assortative mating. If None, no assortative mating is performed. Default is None.
            AM_type (str): Type of assortative mating to perform. If 'phenotypic', uses the trait values directly. If 'genetic', uses the genetic values (only for object's stored traits). Default is 'phenotypic'.

        Returns:
            tuple ((iM, iP), rel_spouses):
            Where:
            - iMs (1D array): Array of length N/2 containing indices of the mothers.
            - iPs (1D array): Array of length N/2 containing indices of the fathers.
            - rel_spouses (2D array): Relationship N*N matrix for spouses, where each element is True if the individuals are spouses and False otherwise.
        '''
        # checks for population size
        if self.N % 2 != 0:
            raise Exception('Population size must be multiple of 2.')
        N2 = self.N // 2
        rel_spouses = np.zeros((self.N,self.N), dtype=bool) # relationship matrix for spouses

        # extracts assortative mating value:
        if AM_r != 0 and AM_trait is not None:
            if type(AM_trait) == np.ndarray:
                AM_values = AM_trait
            elif type(AM_trait) == str:
                if AM_trait not in self.traits:
                    raise Exception(f'Trait {AM_trait} not found in population traits.')
                if AM_type == 'phenotypic':
                    AM_values = self.traits[AM_trait].y
                elif AM_type == 'genetic':
                    AM_values = self.traits[AM_trait].y_['G']
            # standardizes
            AM_values = (AM_values - AM_values.mean()) / AM_values.std()
        else:
            AM_values = np.zeros(self.N)

        # extracts mom (female) and dad (male) indices and shuffles their order
        iMs = np.random.choice(np.where(self.traits['sex'].y == 0)[0], N2, replace=False)
        iPs = np.random.choice(np.where(self.traits['sex'].y == 1)[0], N2, replace=False)

        # computes mate value
        mate_values = AM_r * AM_values + np.random.normal(scale = np.sqrt(1-AM_r**2), size=self.N)
        # sorts mothers and fathers by mate value
        iMs = iMs[np.argsort(AM_values[iMs])]
        iPs = iPs[np.argsort(mate_values[iPs])]    
        # updates relationship matrix for spouses
        rel_spouses[iMs, iPs] = 1
        rel_spouses[iPs, iMs] = 1

        return ((iMs, iPs), rel_spouses)

    def generate_offspring(self, s: Union[float, np.ndarray] = 0,
                           mu: Union[float, np.ndarray] = 0,
                           **kwargs) -> np.ndarray:
        '''
        Pairs up mates and generates offspring for parents' haplotypes. Only works for diploids. Each pair always has two offspring. Recombination rates are extracted from object attributes. Also updates parent-child relationship matrix.
        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float or 1D array): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
            **kwargs: All other arguments (related to assortative mating) are passed to the `_pair_mates` method. See that method for details.
        Returns:
            H (3D array): N*M*P array of offspring haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
            relations (dict): Dictionary containing relationship matrices for the current generation, including 'spouses' and 'parents'.
        '''
        # checks ploidy
        if self.P != 2:
            raise Exception('Offspring generation only works for diploids.')

        # Assortative Mating ####        
        # pairs up mates
        (iMs, iPs), rel_spouses = self._pair_mates()

        # Selection ####
        if isinstance(s, (float, int)):
            s = np.full(self.M, s)
        # computes individuals' breeding weight (assumes linear additive fitness effects)
        W = np.exp( (np.log(1 + self.G * s[None, :])).sum(axis=1) )
        # computes each pair's breeding weight
        W_pair = W[iMs] * W[iPs]
        # computs probability of each parent pair being chosen to mate for one offspring
        P_mate = W_pair / W_pair.sum()
        # determines population size of next generation (currently maintains population size)
        N_offspring = self.N
        # draws indices of parents for each offspring
        i_mate = np.random.choice(np.arange(len(iMs)), size=N_offspring, p=P_mate)
        parents = np.stack((iMs[i_mate], iPs[i_mate]), axis=1)
        # makes relationship matrix for full siblings
        M_mate = np.zeros((N_offspring, len(iMs)), dtype=bool)
        M_mate[np.arange(N_offspring), i_mate] = 1
        rel_fullsibs = M_mate @ M_mate.T
        np.fill_diagonal(rel_fullsibs, 0)

        # Drift + Recombination ####
        # generates variants for which a crossover event happens for each parent of each offspring
        crossover_events = np.random.binomial(n=1, p=self.R.reshape(1, self.M, 1),
                                              size=(N_offspring, self.M, 2))
        # determines the shift in haplotype phase for each parent's haplotype at each variant
        haplo_phase = np.cumsum(crossover_events, axis=1)
        # randomly chooses haplotype to start with for each parent of each offspring
        haplo_k0 = np.random.binomial(n=1, p=1/self.P, size = (N_offspring ,2))
        # adds the starting shift and gets the modulo so that haplotype phase acts as an index
        haplo_ks = (haplo_k0[:, None, :] + haplo_phase) % 2

        rel_parents = np.zeros((N_offspring, self.N), dtype=bool) # relationship matrix for parent-child relationships
        par_idx = np.full((N_offspring, 2), -1, dtype=np.int32) # initializes parent index array
        H = np.empty((N_offspring, self.M, self.P), dtype=int)
        Haplos = np.full_like(H, -1) # initializes haplotype ID array with -1s
        for i in np.arange(N_offspring):
            iM = parents[i,0]
            iP = parents[i,1]
            # extract allele from correct haplotype of each parent
            haploM = self.H[iM, np.arange(self.M), haplo_ks[i, :, 0]]
            haploP = self.H[iP, np.arange(self.M), haplo_ks[i, :, 1]]
            haplos = np.stack((haploM, haploP), axis = 1)
            # shuffles haplotypes around. [EDIT]: nevermind, want to keep consistent maternal vs paternal order
            # chr_order = np.random.choice(self.P, size=self.P, replace=False)
            chr_order = np.arange(self.P)
            haplos = haplos[:,chr_order]
            H[i,:,:] = haplos

            # does same inheritance for haplotype IDs (instead of allele dosages)
            # (yes, I know it's confusing that I am using H/haplo to refer to both)
            HaploM = self.Haplos[iM, np.arange(self.M), haplo_ks[i, :, 0]]
            HaploP = self.Haplos[iP, np.arange(self.M), haplo_ks[i, :, 1]]
            Haplos_i = np.stack((HaploM, HaploP), axis = 1)

            Haplos_i = Haplos_i[:,chr_order]
            Haplos[i,:,:] = Haplos_i

            # updates relationship matrix for parent-child relationships, and stores parent indices
            rel_parents[i, iM] = 1
            rel_parents[i, iP] = 1
            par_idx[i, :] = [iM, iP] # mom is always col 0, dad is always col 1

        # Mutations ####
        if isinstance(mu, (float, int)):
            mu = np.full(self.M, mu)
        mutations = np.random.binomial(n=1, p=mu[None,:,None], size = (N_offspring, self.M, self.P))
        # Apply mutations by flipping alleles (0 to 1 or 1 to 0) based on the mutation matrix
        H = (H + mutations) % 2

        # updates relationship matrix for parent-child relationships
        relations = {'spouses': rel_spouses.astype(np.uint8), # occupies more space than a bool, but cleaner to see
                     'parents': rel_parents.astype(np.uint8),
                     'par_idx': par_idx, # stores actual indices of parents
                     'full_sibs': rel_fullsibs.astype(np.uint8),
                     'household': M_mate.astype(np.uint8)}

        return (H, relations, Haplos)

    def flatten_generations(self, generations: int = 1) -> 'Population':
        '''
        Combines the current generation with the specified number of past generations into a new single population object that is returned. Importantly, it updates the relationship and Pedigree matrices to reflect the relationships in the combined generations.
        Parameters:
            generations (int): Number of past generations to include in the new population object. Default is 1, meaning only the current and the previous generation are combined.
        '''
        # creates a SuperPopulation object with the current generation and the specified number of past generations
        pops = []
        Ns = []
        for i in range(0, generations + 1):
            pops.append(self.past[i]) # at i=0, references itself
            Ns.append(self.past[i].N)
        spop = SuperPopulation(pops)
        # combines the populations together inside the SuperPopulation object
        pops_i = list(range(generations + 1))
        spop.join_populations(pops_i)
        new_pop = spop.pops[-1] # the last population in the SuperPopulation is the combined one
        new_pop.keep_past_generations = self.keep_past_generations
        new_pop.relations = pop.initialize_relations(new_pop.N)
        del new_pop.relations['parents'] # default parents matrix is not accurate here

        # updates relationship matrices in combined population
        # full_sibs
        full_sibs = block_diag(*[pop.relations['full_sibs'] for pop in pops])
        # spouses
        spouses = block_diag(*[pop.relations['spouses'] for pop in pops])
        # parent-child
        parent_child = np.zeros((new_pop.N, new_pop.N), dtype=np.uint8)
        Ns_cumsum = np.cumsum([0] + Ns)
        for gen in range(generations):
            gen_parents = pops[gen].relations['parents']
            i_start = Ns_cumsum[gen]
            i_end = Ns_cumsum[gen + 1]
            j_start = Ns_cumsum[gen + 1]
            j_end = Ns_cumsum[gen + 2]
            parent_child[i_start:i_end, j_start:j_end] = gen_parents
            parent_child[j_start:j_end, i_start:i_end] = gen_parents.T

        # creates accurate Pedigree object for combined population
        # only rel2 can be computed for this initial flattened generation
        Ns_cumsum = np.cumsum([0] + Ns)
        for gen in range(generations+1):
            # fills in main diagonal blocks
            for key, value in pops[gen].ped.rel2.items():
                # need to shift indices based on generation
                if gen > 0:
                    key = (key[0] + Ns_cumsum[gen], key[1] + Ns_cumsum[gen])
                new_pop.ped.rel2[key] = value
            # fills in off diagonal blocks
            if gen == generations:
                continue # no more generations to link to
            # loops through pairs of individuals in current generation and individuals in their parent generation
            for i in range(Ns[gen]):
                for j in range(Ns[gen + 1]):
                    # gets indices (w.r.t. parent generation) of i's parents
                    i_pars = np.flatnonzero( pops[gen].relations['parents'][i,:] )
                    par_keys = ( (i_par, j) for i_par in i_pars ) # prepares keys for get_shortest_path
                    # determines the shortest path between i and j through i's parents
                    shortest_par_path, shortest_path_keys, shortest_path_par_pairs = new_pop.ped.get_shortest_path(pops[gen+1].ped.rel2, par_keys)
                    if shortest_par_path is None:
                        continue # no relationship, don't store anything
                    else:
                        shortest_par_path = self.ped.intern_path(shortest_par_path)

                    # extract the parent index (k: 0/1) of i that is closest to j
                    # in the case that individual j is equally related to both of i's parents, just takes the first one (arbitrary) (denoted by the first [0] below)
                    k = np.where(i_pars == shortest_path_keys[0][0]) # the second [0] denotes we are taking the index of i's parent that is closest to j

                    # extracts the parent index 
                    
                    
                    # extends path to include extra up meiosis (i-->j)
                    path_ij = new_pop.ped.extend_path(shortest_par_path, ups=1, downs=0)
                    path_ij_par_pair = shortest_path_par_pairs
                    # prepares keys for new flattened population
                    key_ij = (i + Ns_cumsum[gen], j + Ns_cumsum[gen + 1])
                    new_pop.ped.rel2[key_ij] = path_ij

                    # fills in reverse direction as well
                    path_ji = new_pop.ped.reverse_path(path_ij)
                    key_ji = (j + Ns_cumsum[gen + 1], i + Ns_cumsum[gen])
                    new_pop.ped.rel2[key_ji] = path_ji
            
        # actually sets relations
        new_pop.relations['full_sibs'] = full_sibs
        new_pop.relations['spouses'] = spouses
        new_pop.relations['parent_child'] = parent_child

        return new_pop

    #######################
    #### Visualization ####
    #######################

    def plot_freq_over_time(self, j_keep: tuple = None,
                            legend=False, last_generations: int = None,
                            summarize: bool = False, quantiles: tuple = (0.25, 0.5, 0.75)):
        '''
        Plots variant allele frequencies over time.
        Parameters:
            j_keep (tuple): Variant indices to include when plotting. Defaults to all variants.
            legend (bool): Whether to include a legend in the plot for each line. Default is False.
            last_generations (int): Number specifying the number of most recent generations to plot. Defaults to all generations since beginning.
            summarize (bool): If true, instead of plotting individual variant trajectories, it plots the mean and specified quantiles of allele frequencies across variants at each generation. Default is False.
            quantiles (tuple): List of quantiles (e.g. 0.99) of allele frequencies across variants at each generation to plot. `summarize` must be set to True. Default is median, lower quartile, and upper quartile.
        '''
        # plots all variants if not specified
        if j_keep is None:
            j_keep = tuple( range(self.M) )
        # uses population's allele frequency history
        ps = self.metric['p']['values']
        # plots all generations if not specified
        if last_generations is None:
            t_start = 0
        else:
            t_start = max(0,ps.shape[0] - last_generations)
        t_keep = tuple( range(t_start, ps.shape[0]))
        # subsets to specified variants
        ts = np.arange(t_start, ps.shape[0])
        ps = ps[np.ix_(t_keep, j_keep)]
        
        # if True, gets mean and quartiles for variants over time, which are plotted instead
        if summarize:
            ps_mean, ps_quantile = pop.summarize_ps(ps, quantiles)
            metrics = np.column_stack((ps_mean, ps_quantile.T))
            aes_line = {'color': ['deepskyblue'] + ['lightskyblue'] * len(quantiles),
                        'ls': ['--'] + [':'] * len(quantiles),
                        'labels': ['Mean'] + [f'{q*100}% percentile' for q in quantiles]}
        else:
            # gets metrics to plot
            metrics = ps
            aes_line = {'color': None, 'ls': '-',
                        'labels': [f'Variant {j_keep[j]}' for j in range(len(j_keep))]}
        # plot aesthetics
        aes = {'title': 'Allele Frequency Trajectories Over Time',
               'xlabel': 'Generation',
               'ylabel': 'Allele Frequency'}

        core.plot_over_time(metrics, ts, aes=aes, aes_line=aes_line, vlines = self.T_breaks, legend=legend)
        
    def plot_LD_matrix(self, LD_matrix: sparse.csr_matrix = None,
                      plot_range: Tuple[int, int] = None, type: str = 'LD',
                      omit_mono: bool = False):
        '''
        Plots LD/correlation between variants.

        Parameters:
            LD_matrix (sparse 2D matrix): Either an LD or correlation M*M matrix of variants. If not provided, uses object's LD/correlation matrix.
            plot_range (tuple): A tuple of length 2 containing the range of variant indices to plot. If not provided, plots all variants.
            type (str): Uses color scheme for either 'LD' (default) or 'corr' matrix.
            omit_mono (bool): Whether variants that are monomorphic (p = 0 or 1) should be skipped over when plotting. Default is False.
        '''
        # chooses between LD and correlation matrix
        if type == 'LD':
            LD_matrix = self.LD_matrix
        elif type == 'corr':
            LD_matrix = self.corr_matrix
        if plot_range is not None:
            start, stop = plot_range
        else:
            start = 0
            stop = LD_matrix.shape[0]

        # Convert the sparse matrix to a dense array
        LD_matrix_dense = LD_matrix[start:stop, start:stop].toarray()
        
        # Skips plotting monomorphic variants
        if omit_mono:
            p = self.p[start:stop]
            j_mono = (p == 0) | (p == 1)
            LD_matrix_dense = np.delete(LD_matrix_dense, j_mono, axis=0)
            LD_matrix_dense = np.delete(LD_matrix_dense, j_mono, axis=1)

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        if type == 'LD':
            plt.imshow(LD_matrix_dense, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        elif type == 'corr':
            plt.imshow(LD_matrix_dense, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar(label='LD Value')
        plt.title(f'{type} Matrix Heatmap')
        plt.xlabel('Variant Index')
        plt.ylabel('Variant Index')
        plt.show()

class Trait:
    '''
    Class for a trait belonging to a Population object.
    '''
    def __init__(self, G: np.ndarray, M_causal: int = None, dist: str = 'normal',
                 var_G: float = 1.0, var_Eps: float = 0.0,
                 random_effects: dict = None):
        '''
        Initializes and generates trait based on variance components.
        Parameters:
            G (2D array): N*M NON-standardized genotype matrix.
            M_causal (int): Number of causal variants (variants with non-zero effect sizes). Default is all variants.
            dist (str): Distribution to draw causal effects from. Options are:
            - 'normal': Normal distribution (default).
            - 'constant': All effect sizes are the same.
            var_G (float): Total expected variance contributed by per-standardized-allele genetic effects. Default is 1.0.
            var_Eps (float): Total expected variance contributed by random noise.
            random_effects (dict): Dictionary of random effects to add to the trait. See `stat.get_random_effects()` for details. Default is None, meaning no random effects are added.
        '''
        # generates causal effects
        effects, _ = stat.generate_causal_effects(G.shape[1], M_causal, var_G, dist)
        self._initialize_effects(G, effects, var_Eps)
        # computes trait
        self.generate_trait(G, random_effects=random_effects)

    @classmethod
    def from_effects(cls, G: np.ndarray, effects: np.ndarray, per_allele: bool = False, var_Eps: float = 0.0,
                 random_effects: dict = None) -> 'Trait':
        '''
        Initializes a trait from a given genotype matrix and genetic effects array.
        Parameters:
            G (2D array): N*M NON-standardized genotype matrix.
            effects (1D array): M-length array of STANDARDIZED effects. Set non-causal effects to 0.
            per_allele (bool): Whether the effects are per-allele. Default is False.
            var_Eps (float): Total expected variance contributed by random noise. Default is 0.0.
            random_effects (dict): Dictionary of random effects to add to the trait. See `stat.get_random_effects()` for details. Default is None, meaning no random effects are added.
        Returns:
            Trait: A new Trait object initialized with the given genotype matrix and effects.
        '''
        trait = cls.__new__(cls)
        # initializes the object with the given haplotype array
        if effects.shape[0] != G.shape[1]:
            raise ValueError("Length of effects must match number of variants in G.")
        # if effects are per-allele, standardize them
        if per_allele:
            G_std = stat.get_G_std_for_effects(G)
            effects = stat.get_standardized_effects(effects, G_std, std2allelic=False)
        trait._initialize_effects(G, effects, var_Eps)
        trait.generate_trait(G, fixed_h2=False, random_effects=random_effects)
        return trait

    @classmethod
    def from_fixed_values(cls, y: np.ndarray) -> 'Trait':
        '''
        Initializes a trait from a given array of fixed trait values. No sub component or effects (e.g. genetic, random effects) are stored.
        Parameters:
            y (1D array): N-length array of trait values.
        '''
        trait = cls.__new__(cls)
        trait.y = y
        trait.type = 'fixed'
        return trait
    
    def _initialize_effects(self, G: np.ndarray, effects: np.ndarray, var_Eps: float = 0.0):
        '''
        Initializes the trait's genetic effects and variance components. See `from_effects` class method for details.
        '''
        self.M = effects.shape[0]
        self.effects = effects
        G_std = stat.get_G_std_for_effects(G)
        self.effects_per_allele = stat.get_standardized_effects(self.effects, G_std, std2allelic=True)
        self.j_causal = np.where(self.effects != 0)[0]  # indices of causal variants
        self.M_causal = len(self.j_causal)
        # variance components
        self.var = {}
        self.var['Eps'] = var_Eps
        self.var['G'] = self.effects.var() * G.shape[1]  # assumes independence!
        self.h2 = self.get_h2_var()

    def generate_trait(self, G: np.ndarray, random_effects: dict = None, fixed_h2: bool = False):
        '''
        Generates/updates trait using stored genetic effects and recomputing other components. Automatically updates object's attributes, instead of returning trait values.
        Parameters:
            G (2D array): N*M NON-standardized genotype matrix.
            random_effects (dict): Dictionary of random effects to add to the trait. See `stat.get_random_effects()` for details. Default is None, meaning no random effects are added.
            fixed_h2 (bool): Whether the variance of the noise component should be updated to maintain the heritability. Genetic component must be non-zero. Default is False.
        '''
        N = G.shape[0]
        # makes empty dictionary for trait components
        self.y_ = {}
        # genetic component (using per-allele effects)
        self.y_['G'] = stat.compute_genetic_value(G, self.effects_per_allele)

        # random effects components
        if random_effects is not None:
            for i in range(len(random_effects['Z'])):
                Zu = random_effects['Z'][i] @ random_effects['u'][i]
                self.y_[random_effects['name'][i]] = Zu

        var_Eps = self.var['Eps']
        # computes non-noise component of trait by adding up all 'y_' components except for 'Eps'
        y_nonEps = np.sum([self.y_[key] for key in self.y_.keys() if key != 'Eps'], axis=0)
        # recomputes non-noise component to get needed var_Eps to maintain heritability
        if fixed_h2:
            var_Eps = (y_nonEps.var() / self.h2) - y_nonEps.var()

        # random noise component
        self.y_['Eps'] = stat.generate_noise_value(N,var_Eps)

        # computes actual trait as additive of individual components
        self.y = y_nonEps + self.y_['Eps']

        # sets trait type has being a composite of multiple components
        self.type = 'composite'

    def get_h2_true(self) -> float:
        '''
        Returns the true narrow-sense heritability of the trait, which is variance of the genetic component divided by the variance of the trait.
        Returns:
            h2_true (float): True heritability of the trait.
        '''
        h2_true = self.y_['G'].var() / self.y.var()
        return h2_true
    
    def get_h2_var(self) -> float:
        '''
        Returns the expected heritability based on the variance components of the trait. This is computed as the variance of the genetic component divided by the sum of variance components. This is likely to not be accurate if trait components are not independent. For a more accurate estimate, use `get_h2_true()`.
        Returns:
            h2 (float): Heritability of the trait.
        '''
        h2 = self.var['G'] / np.sum(list(self.var.values()))
        return h2
    
    def get_components_matrix(self, exclude = None, include_y = True) -> np.ndarray:
        '''
        Returns a matrix of the trait components, where each column is a component and each row is an individual. Can exclude certain components if desired.
        Parameters:
            exclude (list): List of component names to exclude from the matrix. Default is None, meaning no components are excluded.
            include_y (bool): Whether to include the overall trait values as the last column in the matrix. Default is True.
        Returns:
            component_matrix (2D array): N*C matrix of trait components, where N is the number of individuals and C is the number of components (including overall trait if `include_y` is True).
        '''
        components = []
        for key in self.y_.keys():
            if exclude is not None and key in exclude:
                continue
            components.append(self.y_[key])
            
        if include_y:
            components.append(self.y)

        component_matrix = np.column_stack(components)
        return component_matrix

    @classmethod
    def concatenate_traits(cls, traits: list, G: np.ndarray) -> 'Trait':
        '''
        Concatenates multiple Trait objects from multiple populations into a single Trait object representing that population. Assumes that the Trait objects represent the same trait, and that they have the same per-allele genetic effects.
        Parameters:
            traits (list): List of Trait objects to concatenate.
            G (2D array): N*M NON-standardized genotype matrix with the joined population's genotypes. Used for converting per-allele effects to standardized effects.
        Returns:
            trait_new (Trait): A new Trait object containing the concatenated trait.
        '''
        # if trait has components, then those effects (e.g. allelic effects)needs to be carried over
        if traits[0].type == 'composite':
            # pulls per-allele effects from first population's Trait object   
            effects_per_allele = traits[0].effects_per_allele
            # creates placeholder Trait object (attributes will largely be overwritten)
            trait_new = cls.from_effects(G, effects=effects_per_allele, per_allele=True)
        elif traits[0].type == 'fixed':
            trait_new = cls.from_fixed_values(np.array([])) # creates empty Trait object to be filled in below
        
        # concatenates trait components
        trait_new.y = np.concatenate([trait.y for trait in traits])
        if trait_new.type == 'composite':
            for component in traits[0].y_.keys():
                # assumes all traits have the same components
                trait_new.y_[component] = np.concatenate([trait.y_[component] for trait in traits])
            # updates variance components
            trait_new.var['Eps'] = trait_new.y_['Eps'].var()
            trait_new.h2 = trait_new.get_h2_var()

        return trait_new

    def index_trait(self, i_keep: np.ndarray, G: np.ndarray, G_already_indexed: bool = False) -> 'Trait':
        '''
        Returns a Trait object that contains only the trait data for the specified individuals' indices.
        Parameters:
            i (1D array): Array of length N_new containing indices of individuals to keep in the trait.
            G (2D array): N*M NON-standardized genotype matrix. Can either be the original full genotype matrix or the version of it that has already been indexed to only include the individuals in `i`, depending on the `G_already_indexed` parameter.
            G_already_indexed (bool): Whether the genotype matrix `G` has already been indexed to only include the individuals in `i`. If True, `G` should be a N_new*M matrix. If False, `G` should be the original full genotype matrix. Default is False.
        Returns:
            Trait: A new Trait object containing only the specified indices of the trait.
        '''
        if not G_already_indexed:
            # if G is not already indexed, index it
            G = G[i_keep, :]
        trait_new = copy.deepcopy(self)
        # updates standardized effects
        G_std = stat.get_G_std_for_effects(G)
        trait_new.effects = stat.get_standardized_effects(self.effects_per_allele, G_std, std2allelic=False)
        # indexes trait components
        trait_new.y = self.y[i_keep]
        for component in self.y_.keys():
            # assumes all traits have the same components
            trait_new.y_[component] = self.y_[component][i_keep]
        # updates variance components
        trait_new.var['Eps'] = trait_new.y_['Eps'].var()
        trait_new.h2 = trait_new.get_h2_var()

        return trait_new
        
class SuperPopulation:
    '''
    Class for a superpopulation, which contains multiple populations. Allows for multiple populations to be simulated forward in time together.
    '''

    #######################
    #### Intialization ####
    #######################
    def __init__(self, pops: Union[Population, list]):
        '''
        Initializes a superpopulation with a list of populations.
        Parameters:
            pops (list): List of Population objects to include in the superpopulation. Can also just be a single Population object.
        '''
        if isinstance(pops, Population):
            # if only a single population is passed, convert it to a list
            pops = [pops]
        self.pops = pops.copy() # makes a copy of the list of populations
        # initializes basic attributes
        self.era = 0 # will get updated to 1 by _update_era() below
        # creates active vector
        self.active = [True] * len(pops) # active populations as boolean
        self._update_era() # initializes active indices and history
        # creates lineage graph (adjacency matrix) for populations
        self._expand_graph()
    
    def _update_era(self):
        ''' Updates the list of active population indices based on the current state of the `active` attribute, the current era, and the history of active populations (as boolean matrix).'''
        # gets indices of active populations
        self.active_i = [i for i, active in enumerate(self.active) if active]  # active indices

        # updates history of active populations (and updates era)
        self.era += 1        
        active_history = np.zeros((self.era, len(self.pops)), dtype=bool)
        if hasattr(self, 'active_history'):
            active_history[:(self.era-1), :self.active_history.shape[1]] = self.active_history
        
        active_history[self.era-1, :] = self.active
        self.active_history = active_history

    ##################################
    #### Updating population list ####
    ##################################
    def add_population(self, pops: Union[Population, list], active_new: Union[bool, list] = True,
                       update_era: bool = True):
        '''
        Adds one or more new populations to the superpopulation.
        Parameters:
            list (Population or list): Population object to add. If a list is passed, it is assumed to be a list of Population objects, all of which are added in the order provided.
            active_new (bool or list): Whether the new populations should be active. If a single boolean is passed, it applies to all new populations. If a list is passed, it should have the same length as the number of new populations and specifies whether each population is active. Default is True, meaning all new populations are active.
            update_era (bool): Whether to update the era after inactivating the populations. Default is True.
        '''
        if isinstance(pops, Population):
            # if only a single population is passed, convert it to a list
            pops = [pops]
        
        self.pops.extend(pops)
        # updates active populations
        if isinstance(active_new, bool):
            # if only a single boolean is passed, apply it to all new populations
            active_new = [active_new] * len(pops)
        elif len(active_new) != len(pops):
            raise ValueError("Length of active_new must match the number of new populations.")
    
        # updates active status
        self.active.extend(active_new)
        if update_era: 
            self._update_era() # updates era, active indices, and history
        # updates lineage graph
        self._expand_graph()
        
    def inactivate_population(self, pop_i: Union[int, list], update_era: bool = True):
        '''
        Inactivates one or more populations from the superpopulation.
        Parameters:
            pop_i (int or list): Index of the population to inactivate. If a list is passed, it is assumed to be a list of indices of populations to inactivate.
            update_era (bool): Whether to update the era after inactivating the populations. Default is True.
        '''
        if isinstance(pop_i, int):
            pop_i = [pop_i]
        for i in pop_i:
            self.active[i] = False
        if update_era: 
            self._update_era() # updates era, active indices, and history

    def activate_population(self, pop_i: Union[int, list], update_era: bool = True):
        '''
        Activates one or more populations in the superpopulation.
        Parameters:
            pop_i (int or list): Index of the population to activate. If a list is passed, it is assumed to be a list of indices of populations to activate.
            update_era (bool): Whether to update the era after activating the populations. Default is True.
        '''
        if isinstance(pop_i, int):
            pop_i = [pop_i]
        for i in pop_i:
            self.active[i] = True
        if update_era: 
            self._update_era() # updates era, active indices, and history

    def join_populations(self, pop_i: list):
        '''
        Joins multiple populations into a single population. Inactivates the original populations and creates a new population from the merged haplotypes. The new population is added to the superpopulation as an active population.
        Parameters:
            pop_i (list): List of indices of populations to join.
        '''
        if len(pop_i) < 2:
            raise ValueError("Must specify at least two populations to join.")
        
        # merges haplotypes of specified populations
        H = np.concatenate([self.pops[i].H for i in pop_i], axis=0)
        # creates new population from merged haplotypes
        new_pop = Population.from_H(H, keep_past_generations=0)
        # adds Trait objects by concatenating them, assumes the first population has all traits
        for name in self.pops[pop_i[0]].traits.keys():
            # concatenates traits from all populations being joined
            traits = [self.pops[i].traits[name] for i in pop_i]
            new_pop.traits[name] = Trait.concatenate_traits(traits, new_pop.G)
        
        # updates superpopulation
        self.add_population(new_pop, active_new=True, update_era=False)
        self.inactivate_population(pop_i)

        # updates graph
        for i in pop_i:
            # updates graph to reflect that the populations are now joined
            self.graph[i, len(self.pops)-1] = 1

    def split_population(self, pop_i: int, N_new: Union[int, list] = 2):
        '''
        Splits a population into two or more populations. This is done by randomly drawing individuals from the source population. The new populations are created from the haplotypes of the original population. The original population is inactivated and the new populations are added to the superpopulation as active populations.
        Parameters:
            pop_i (int): Index of the population to split.
            N_new (int or list): Number of individuals in each new population. If a single integer is passed (Default = 2), it is treated as the number of equally-sized populations to split into. If a list is passed, it should have the same length as the number of new populations and specifies the number of individuals in each population. The list must sum to less than or equal to the number of individuals in the original population.
        '''
        source = self.pops[pop_i]
        # checks if population can be split
        if isinstance(N_new, int):
            # if only a single integer is passed, split into that many equally-sized populations
            N_new = [source.N // N_new] * N_new # rounds down to nearest integer
        elif len(N_new) < 2:
            raise ValueError("Must specify at least two new populations to split into.")
        elif sum(N_new) > source.N:
            raise ValueError("Sum of N_new must be less than or equal to the number of individuals in the original population.")
        K = len(N_new)  # number of new populations
        # shuffles indices of individuals in the source population
        i_shuffled = np.arange(source.N)
        np.random.shuffle(i_shuffled)  # shuffles haplotypes
        i_start = 0
        new_pops = []
        for i in range(K):
            # gets indices of individuals for the new population
            i_end = i_start + N_new[i]
            i_new = i_shuffled[i_start:i_end]
            # creates new population from the haplotypes of the original population
            H_new = source.H[i_new, :, :]
            new_pop = Population.from_H(H_new)
            # updates traits
            for name, trait in source.traits.items():
                # concatenates traits from all populations being joined
                new_pop.traits[name] = trait.index_trait(i_new, source.G, G_already_indexed=False)
            
            new_pops.append( new_pop )
            # updates next index to start pulling from
            i_start = i_end
        # adds new population to the superpopulation
        new_pops_i = np.arange(len(self.pops), len(self.pops) + K)
        self.add_population(new_pops, active_new=True, update_era=True)
        self.inactivate_population(pop_i)
        # updates graph
        for i in new_pops_i:
            # updates graph to reflect that the populations are now split
            self.graph[pop_i, i] = 1

    #######################
    #### Lineage Graph ####
    #######################
    def _expand_graph(self):
        '''
        Expands lineage graph to match current number of populations. Can also be used to initialize the graph if it doesn't exist yet.
        '''
        graph = np.zeros((len(self.pops), len(self.pops)), dtype=int)
        # updates graph attribute
        if hasattr(self, 'graph'):
            if len(self.pops) < len(self.graph):
                raise ValueError("List of populations has shrunk! This shouldn't happen.")
            graph[:len(self.graph), :len(self.graph)] = self.graph
        self.graph = graph

    ################
    #### Traits ####
    ################
    def add_trait(self, name: str, per_allele_p_pop: int = None, **kwargs):
        '''
        Adds a trait to all active populations in the superpopulation.
        Parameters:
            name (str): Name of the trait to add.
            per_allele_p_pop (int): The index of the population from which to pull allele frequencies in order to generate per-allele effect sizes. The method first generates standardized per-allele effects, but for biological realism, effects are fixed across populations by scaling them to be per-allele effect sizes. This requires having a set of allele frequencies (or more precisely, genotype standard deviations) to scale effects by, which can be extracted from the specified population index. If not provided (Default), the method will join the active populations together and get the standard deviation of the genotype matrix across them.
            **kwargs: All arguments are analogous to the `Trait` constructor method. See that method for details. However, the per-allele genetic effects are shared across all populations. For each parameter except `var_G` and `M_causal`, if a Python list is passed, it is assumed to be a list of arguments for each population in the superpopulation. If a list isn't passed, it is used for all populations.
        '''
        # generates genetic effects
        M = self.pops[0].M  # assumes all populations have the same number of variants
        M_causal = kwargs.get('M_causal', M)
        var_G = kwargs.get('var_G', 1.0)
        effects, _ = stat.generate_causal_effects(M, M_causal, var_G)
        if per_allele_p_pop is not None:
            # gets standard deviation of genotype matrix from specified population
            G_std = stat.get_G_std_for_effects(self.pops[per_allele_p_pop].G)
        else:
            # gets average standard deviation of genotype matrix across all active populations
            G_std = np.mean([stat.get_G_std_for_effects(pop.G) for pop, is_active
                             in zip(self.pops, self.active) if is_active], axis=0)
        
        effects_per_allele = stat.get_standardized_effects(effects, G_std, std2allelic=True)

        # iterates through each active population
        for i, pop_i in enumerate(self.active_i):
            pop = self.pops[pop_i]
            pop_kwargs = core.get_pop_kwargs(i, **kwargs)
            # Remove keys not accepted by add_trait_from_effects
            pop_kwargs.pop('var_G', None)
            pop_kwargs.pop('M_causal', None)
            # adds trait to population using pre-computed effects
            pop.add_trait_from_effects(name=name, effects=effects_per_allele, per_allele=True, **pop_kwargs)

    ####################
    #### Simulating ####
    ####################
    def simulate_generations(self, **kwargs):
        '''
        Simulates generations for all active populations in the superpopulation. 
        Parameters:
            **kwargs: All arguments are passed to the `simulate_generations()` method of each Population object. See that method for details. For each parameter, if a Python list is passed, it is assumed to be a list of arguments for each population in the superpopulation. If a list isn't passed, it is used for all populations.
        '''
        # iterates through each active population
        for i, pop_i in enumerate(self.active_i):
            pop = self.pops[pop_i]
            # creates kwargs list for each population
            pop_kwargs = {}
            # simulates generations for the population using population-specific kwargs
            pop_kwargs = core.get_pop_kwargs(i, **kwargs)
            pop.simulate_generations(**pop_kwargs)

    #######################
    #### Visualization ####
    #######################
    def print_attributes(self, attribute: str, only_active: bool = True):
        '''
        Prints the attributes of all populations in the superpopulation.
        Parameters:
            only_active (bool): Whether to only print attributes of active populations. Default is True.
        '''
        # iterates through each population
        for i, pop in enumerate(self.pops):
            if only_active and not self.active[i]:
                continue
            print(f'Population {i}:')
            print(getattr(pop, attribute, 'Attribute not found.'))


# the following classes are related to storing relatedness/pedigree information

# data types
PedPath = Tuple[int, ...] # stores chain of meiosis up/down, modified from [Williams et al. 2025 Genetics]
PedKey2 = Tuple[int,int] # stores pair of individual indices (i,j)
PedParPos = Tuple[int,int] # stores pair of parental indices (k,l: 0/1)
PedKey4 = Tuple[int,int,int,int] # stores pair of individual indices + their parents (i,j,k,l)

@dataclass
class PedPairwise:
    path: PedPath
    parent_pair: Tuple[PedParPos, ...]  # list of parent pair(s) leading to this relationship

class Pedigree:
    def __init__(self, N: int, par_idx: np.ndarray = None, par_Ped: 'Pedigree' = None):
        '''
        Initializes a Pedigree object.
        Parameters:
            N (int): population size of current generation
            par_idx (N*2 array): stores the indices of individual i's two parents. These parental indices should correspond to the parent generation's indices. Default is None (e.g. founding generation).
            par_Ped (Pedigree): pointer to the parent generation's Pedigree object. Default is None (e.g. founding generation).
        '''
        # pairwise and pairwise+parent dictionaries
        self.rel2: Dict[PedKey2, PedPath] = {}
        self.N = N # population size
        self.par_idx = par_idx if par_idx is not None else None
        self.par_Ped = par_Ped if par_Ped is not None else None

        # maps PedPath to a canonical PedPath for storage efficiency
        self._paths: Dict[PedPath, PedPath] = {}

        # fills in rel2 with self-relationships (denoted by () )
        self.fill_rel2_self()

    def intern_path(self, path: PedPath) -> PedPath:
        '''
        Return the canonical tuple object for this PedPath. Small wrapper around dict.setdefault for clarity.
        '''
        return self._paths.setdefault(path, path)

    def extend_path(self, path: PedPath, ups: tuple = (), downs: tuple = ()) -> PedPath:
        '''
        Given a path, extends it by adding the specified number of meioses up and down. Also interns the path before returning.
        Parameters:
            path (PedPath): The path to extend.
            ups (tuple): Tuple of meioses to add going up the pedigree (added at beginning of path). Should be positive. Default is (), i.e. no meioses added.
            downs (tuple): Tuple of meioses to add going down the pedigree (added at end of path). Should be negative. Default is (), i.e. no meioses added.
        Returns:
            PedPath: The extended path.
        '''
        # formats tuples properly if length 1 integer is passed
        if isinstance(ups, int):
            ups = (ups,)
        if isinstance(downs, int):
            downs = (downs,)
        extended_path = ups + path + downs
        return self.intern_path(extended_path)
    
    @staticmethod
    def get_closest_path(rel2: Dict, keys: Tuple) -> Tuple[Optional[PedPath], List, Optional[List]]:
        '''
        Given a dictionary of relationships and a list of keys, returns the closest path among the keys.
        Parameters:
            rel2 (Dict): Dictionary of relationships (rel2 attribute in a Pedigree object).
            keys (Tuple): Tuple of keys to check.
        Returns:
            tuple ((closest_path, closest_path_keys)):
            Where:
            - closest_path (PedPath): The closest path found among the keys.
            - closest_path_keys (List): List of keys that correspond to the closest path.
        '''
        closest_path = None # PedPath object
        closest_path_keys = []
        for key in keys:
            path = rel2.get(key)
            # skips if pair is unrelated
            if path is None:
                continue
            
            # series of checks to determine if this path is the closest yet
            if closest_path is None:
                closest_path = path
                closest_path_keys = [key]
                continue
            # first check: path cannot be longer than closest path so far
            if len(path) > len(closest_path):
                continue
            # second check: path cannot have less (+3) entries than closest path so far 
            if path.count(3) < closest_path.count(3):
                continue
            # third check: path cannot have less (-1) entries than closest path so far
            if path.count(-1) < closest_path.count(-1):
                continue
            # if all these checks are passed, we set this as the closest path
            # if this path is identical to the previously stored closest path, we append the key
            if path == closest_path:
                closest_path_keys.append(key)
            else:
                closest_path = path
                closest_path_keys = [key]
            
        return closest_path, closest_path_keys

    # def compress_rel4_to_rel2(self, i: int, j: int) -> Optional[PedPairwise]:
    #     '''
    #     Given two individuals' parents, returns the relationship relative to the parent pair with the shortest path. Specifically, it is the path of i to j.
    #     Parameters:
    #         i (int): Index of individual i.
    #         j (int): Index of individual j.
    #     Returns:
    #         Optional[PedPairwise]: The relationship between individuals i and j, or None if they are unrelated
    #     '''
    #     # there are four possible parent pairs (k,l) for the offspring pair (i,j)
    #     parent_pair_keys = [(i,j,k,l) for k in (0,1) for l in (0,1)]
    #     # extracts shortest path among parent pairs from rel4 dictionary
    #     shortest_path, shortest_path_keys, _ = self.get_shortest_path(self.rel4, parent_pair_keys)
        
    #     # if all parent pairs are unrelated, then so is the offspring pair
    #     if shortest_path is None:
    #         return None
    #     else:
    #         # we only care about the parent pairs (k,l) for the offspring pair (i,j)
    #         shortest_path_pairs = [ (k,l) for (_,_,k,l) in shortest_path_keys ]
    #         # extends the shortest parent path to the offspring pair path
    #         i2j_path = self.extend_path(shortest_path, ups=1, downs=1)
    #         return PedPairwise(path=i2j_path, parent_pair=tuple(shortest_path_pairs))
    
    def reverse_path(self, path: PedPath) -> PedPath:
        '''
        Reverse a PedPath (flip order and sign) and intern it. Example: (1,-3,-1) -> (1,3,-1). Also interns the path before returning.
        Parameters:
            path (PedPath): The path to reverse.
        Returns:
            PedPath: The reversed path.
        '''
        reversed_path = tuple(-step for step in path[::-1])
        return self.intern_path(reversed_path)

    # def reverse_pairwise(self, pw_ij: PedPairwise) -> PedPairwise:
    #     '''
    #     Reverse a PedPairwise: flip path and flip parent positions (k,l) -> (l,k).
    #     Parameters:
    #         pw_ij (PedPairwise): The PedPairwise to reverse.
    #     Returns:
    #         PedPairwise: The reversed PedPairwise.
    #     '''
    #     pw_ji_path = self.reverse_path(pw_ij.path)
    #     pw_ji_pairs = tuple((l, k) for (k, l) in pw_ij.parent_pair)
    #     pw_ji = PedPairwise(path=pw_ji_path, parent_pair=pw_ji_pairs)
    #     return pw_ji

    def fill_rel2_self(self):
        '''
        Fills in the rel2 dictionary with self-relationships only.
        '''
        self_path = self.intern_path( () ) # empty path for self
        for i in range(self.N):
            self.rel2[(i,i)] = self_path

    def construct_rel2(self):
        '''
        Constructs a dictionary with the relationship paths between every related individual in the current population. Pedigree object must have par_idx and par_Ped attributes. See __init__(). For individual i and j, the shortest relationship path between the 4 pairs of parents of i and j is used, and then extended by one meiosis up and down to get the relationship path between i and j. This can erase inbreeding information in the current generation. Only entries i>j are explicitly computed, since the reverse relationships are automatically filled in.
        '''
        # iterates over all pairs of individuals (i > j)
        for i in range(self.N):
            for j in range(i+1, self.N):
                i_pars = np.array(self.par_idx[i, :])  # (mom ID, dad ID)
                j_pars = np.array(self.par_idx[j, :])  # (mom ID, dad ID)
                # gets shortest path and the keys that produced it
                keys = [ (i_pars[k], j_pars[l]) for k in (0,1) for l in (0,1) ]
                path_pars, pargen_keys = self.get_closest_path(self.par_Ped.rel2, keys)

                # if all parent pairs are unrelated, then so is the offspring pair
                if path_pars is None:
                    continue
                # determines which parent(s) (0 or 1) for i and j produced the closest path
                i_par_closest = [(np.where(i_pars == i_k)[0][0]) for (i_k,j_l) in pargen_keys]
                j_par_closest = [(np.where(j_pars == j_l)[0][0]) for (i_k,j_l) in pargen_keys]
                if 0 in i_par_closest and 1 in i_par_closest:
                    up_sex =  3 # if both parents of i have a closest path, use +3 encoding
                else:
                    up_sex = i_par_closest[0] + 1 # otherwise, use the parent that produced the closest path (incremented by 1 for encoding)
                if 0 in j_par_closest and 1 in j_par_closest:
                    down_sex = -3 # if both parents of j have a closest path, use -3 encoding
                else:
                    down_sex = -(j_par_closest[0] + 1) # otherwise, use the parent that produced the closest path (incremented by 1 and then flipped signfor encoding)

                # extends closest parental path by adding sex-encoded up/down meioses
                path_ij = self.extend_path(path_pars, ups=(up_sex,), downs=(down_sex,) )
                # passes paths through intern_path() to ensure canonical storage
                path_ij = self.intern_path(path_ij)
                # constructs the reverse relationship of j to i
                path_ji = self.reverse_path(path_ij) # (path is interned inside function)

                # stores both relationships
                self.rel2[(i,j)] = path_ij
                self.rel2[(j,i)] = path_ji

        # handles self-relationships
        self.fill_rel2_self()

    # def fill_rel2_from_rel4(self, force: bool = False):
    #     '''
    #     Fills in the object's rel2 dictionary from its rel4 dictionary by compressing each offspring pair's relationship to the shortest parent pair relationship. Only entries i>j are explicitly computed, since the reverse relationships are automatically filled in.
    #     Parameters:
    #         force (bool): Whether to overwrite existing entries in the rel2 dictionary. Default is False. When False, overwrites both (i,j) and (j,i) if (i,j) is not present.
    #     '''
    #     # iterates over all pairs of individuals (i > j)
    #     for i in range(self.N):
    #         for j in range(i+1, self.N):
    #             # checks if relationship already exists
    #             if not force and (i,j) in self.rel2:
    #                 continue
    #             # gets the compressed relationship of i to j
    #             ped_ij = self.compress_rel4_to_rel2(i, j)
    #             # if no relationship exists (unrelated), skips
    #             if ped_ij is  None:
    #                 continue
                
    #             # passes paths through intern_path to ensure canonical storage
    #             ped_ij.path = self.intern_path(ped_ij.path)

    #             # constructs the reverse relationship of j to i
    #             ped_ji = self.reverse_pairwise(ped_ij) # (path is interned inside function)
                
    #             # stores both relationships
    #             self.rel2[(i,j)] = ped_ij
    #             self.rel2[(j,i)] = ped_ji
        
    #     # handles self-relationships
    #     self.fill_rel2_self()

    # def construct_rel4(self, par_idx: np.ndarray, par_ped: 'Pedigree'):
    #     '''
    #     Constructs the rel4 dictionary for the current generation based on the Population.relations['par_idx'] matrix from the Population object and the Pedigree of the parents.
    #     Parameters:
    #         par_idx (2D array): N x 2 array of parent indices for each offspring in the current population. Each row corresponds to an offspring, with the first column being the mother index and the second column being the father index. The index is given in the context of the parent population.
    #         par_ped (Pedigree): The Pedigree object of the parent population.
    #     '''
    #     N_offspring = par_idx.shape[0] # this should be the same as self.N!
    #     if N_offspring != self.N:
    #         raise ValueError("Number of offspring in rel_parents does not match self.N.")
        
    #     # stores unique offspring parent pairs
    #     parent_pairs = {}
    #     for i in range(N_offspring):
    #         i_par_indices = tuple( par_idx[i, :]) # (mom ID, dad ID)
    #         if len(i_par_indices) != 2:
    #             raise ValueError("Each offspring must have exactly two parents.")
    #         if i_par_indices not in parent_pairs:
    #             parent_pairs[i_par_indices] = [i]
    #         else:
    #             parent_pairs[i_par_indices].append(i)

    #     # nested loop galore, i know
    #     # nested loop through each parent pair combination
    #     parent_pairs_items = list(parent_pairs.items())
    #     for i_pars_idx in range(len(parent_pairs_items)):
    #         i_pars = parent_pairs_items[i_pars_idx][0]
    #         for j_pars_idx in range(i_pars_idx, len(parent_pairs_items)):
    #             j_pars = parent_pairs_items[j_pars_idx][0]
    #             # gets each parent pairs' offspring
    #             i_offspring = parent_pairs[i_pars]
    #             j_offspring = parent_pairs[j_pars]
    #             # fills in rel4 for each offspring pair
    #             # loops through each cross-parent pair
    #             for k in (0,1):
    #                 for l in (0,1):
    #                     # gets path objectbetween the two cross-parents
    #                     ik_jl_path_obj = par_ped.rel2.get( (i_pars[k], j_pars[l]) )
    #                     # doesn't fill in anything if parents are unrelated
    #                     if ik_jl_path_obj is None:
    #                         continue
    #                     # extracts (canonical) path
    #                     ik_jl_path = self.intern_path(ik_jl_path_obj.path)
    #                     # stores parent path in rel4 for each offspring pair
    #                     for i in i_offspring:
    #                         for j in j_offspring:
    #                             if i == j:
    #                                 continue # skips self-relationships when the two pairs are actually just the same (e.g. siblings). Still computes pairwise relationships between siblings
    #                             # stores path for offspring pair and its reverse
    #                             self.rel4[(i,j,k,l)] = ik_jl_path
    #                             self.rel4[(j,i,l,k)] = self.reverse_path(ik_jl_path)