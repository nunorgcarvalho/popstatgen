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
from typing import Tuple, Union
from scipy import sparse
import copy
import inspect

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
                 seed: int = None):
        '''
        Initializes a population, simulating initial genotypes from specified allele frequencies.
        Parameters:
            N (int): Population size of individuals (not haplotypes).
            M (int): Total number of variants in genome.
            P (int): Ploidy of genotpes. Default is 2 (diploid).
            p_init (float or array): Initial allele frequency of variants. If only a single value is provided, it is treated as the initial allele frequency for all variants. Alternatively, can be an array of length M for variant-specfic allele frequencies. If not provided, default is uniform distribution of allele frequencies between 0.05 and 0.95.
            keep_past_generations (int): Number of past generations to keep in the object. Default is 1, meaning the past generation is kept (on top of the current generation).
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
        self._initialize_H(H, keep_past_generations=keep_past_generations)
    
    @classmethod
    def from_H(cls, H: np.ndarray, keep_past_generations: int = 1):
        '''
        Initializes a population from a given haplotype array.
        Parameters:
            H (3D array): N*M*P array of haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
            keep_past_generations (int): Number of past generations to keep in the object. Default is 1, meaning the past generation is kept (on top of the current generation).
        Returns:
            Population: A new Population object initialized with the given haplotype array.
        '''
        # creates new instance of class
        pop = cls.__new__(cls)
        # initializes the object with the given haplotype array
        pop._initialize_H(H, keep_past_generations=keep_past_generations)
        return pop

    def _initialize_H(self, H: np.ndarray, keep_past_generations: int = 1):
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
        self.K = np.diag(np.ones(self.N)) # kinship matrix (initially identity)
        self.keep_past_generations = keep_past_generations # how many past generations to keep in memory
        self.past = [self]
        for _ in range(keep_past_generations):
            self.past.append(None) # initializes past generations' objects as None

        # further attributes
        self._update_obj(H=H, update_past=False)
        self.ps = np.expand_dims(self.p, axis=0)

        # defines metrics
        self.metric = {}
        self._define_metric('p', pop.compute_freqs, shape = [self.M], # allele frequency
                            P = self.P)
        self._initialize_metrics(G = self.G)

    ###################################
    #### Storing object attributes ####
    ###################################

    def _update_obj(self, H: np.ndarray = None, update_past: bool = True):
        '''
        Update the population object's attributes.
        Parameters:
            H (3D array): Haplotype array.
            update_past (bool): Whether to update the memory of past generations. Default is True.
        '''
        if update_past:
            self._update_past()
        if H is not None:
            self.H = H
            self.G = pop.make_G(self.H)
            self.p = pop.compute_freqs(self.G, self.P)
            self.X = pop.standardize_G(self.G, self.p, self.P, impute=True, std_method='observed')
        
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
            raise Exception('Must have pre-computed neighbor_matrix. Use `Population.get_neighbor_matrix()`.')
        self.corr_matrix = pop.compute_corr_matrix(self.X, self.neighbor_matrix)
        self.LD_matrix = pop.compute_LD_matrix(self.corr_matrix)

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
    
    def update_traits(self, fixed_h2: bool = True, traits: list = None):
        '''
        Updates all traits by generating based on the current genotype matrix. Random noise components are re-generated. Causal genetic effects remain fixed.

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
            trait.generate_trait(self.G, fixed_h2=fixed_h2)
            # computes true heritability
            trait.h2_true = trait.get_h2_true()
            # adds to running heritability history
            t_last = trait.h2_trues.shape[0]
            h2_add = np.full(self.t - t_last + 1, np.nan)
            h2_add[-1] = trait.h2_true
            trait.h2_trues = np.concatenate( (trait.h2_trues, h2_add) )

    #######################################
    #### Analysis of object attributes ####
    #######################################

    def get_fixation_t(self, ps: np.ndarray = None) -> np.ndarray:
        '''
        For each variant, finds *first* generation (returned as an index) for which the allele frequency is 0 (loss) or 1 (fixation). A value of -1 means the variant never got fixed

        Parameters:
            ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time. Defaults to object's allele frequency history.
        
        Returns:
            t_fix (1D array): Array of length M with the first generation (as an index) for which the respective allele was lost or fixed. If the allele was not fixed by the most recent simulation, a -1 is returned.
        '''
        # uses population's allele frequency history if not specified
        if ps is None:
            ps = self.ps
        # gets mask of whether frequency is 0 or 1
        ps_mask = np.any((ps == 0, ps == 1), axis=0)
        # finds first instance of True for each variant
        t_fix = np.where(ps_mask.any(axis=0), ps_mask.argmax(axis=0), -1)        
        return t_fix
    
    def summarize_ps(self, ps: np.ndarray = None, quantiles: tuple = (0.25, 0.5, 0.75)) -> tuple[np.ndarray, np.ndarray]:
        '''
        Returns the mean as well as the specified quantiles of variants across each generation.

        Parameters:
            ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time. Defaults to object's allele frequency history.
            quantiles (tuple): List of quantiles (e.g. 0.99) of allele frequencies across variants at each generation to plot. `summarize` must be set to True. Default is median, lower quartile, and upper quartile.
        
        Returns:
            tuple ((ps_mean, ps_quantile)):
            Where:
            - ps_mean (1D array): Array of length T (where T is the total number of generations) of mean allele frequency at each generation.
            - ps_quantile (2D array): K*T matrix (where K is the number of quantiles specified) of allele frequency for each quantile at each generation.
        '''
        # uses population's allele frequency history if not specified
        if ps is None:
            ps = self.ps
        # computes mean allele frequency over time
        ps_mean = self.ps.mean(axis=1)
        # computes quantiles over time
        ps_quantile = np.quantile(self.ps, quantiles, axis=1)
        return (ps_mean, ps_quantile)
    
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
            trait_updates (bool): Whether to update traits after each generation. Default is False, meaning that traits are only updated at the end of the simulation.
            **kwargs: All other arguments are passed to the `next_generation` or `generate_offspring` methods. See those methods for details.
        '''
        # preps metrics for new generations
        self._prep_metrics(generations)
        # keeps track of allele frequencies over generations if specified
        previous_gens = self.ps.shape[0]
        ps = np.full( (previous_gens + generations, self.M), np.nan)
        ps[0:previous_gens,] = self.ps
        
        # loops through each generation
        for t in range(generations):
            if related_offspring:
                H = self.generate_offspring(**kwargs)
            else:
                H = self.next_generation(**kwargs)
            # updates objects and past
            self._update_obj(H=H)
            # records metrics
            ps[previous_gens + t,] = self.p
            self._update_temp_metrics(t, G=self.G)
            if trait_updates:
                self.update_traits(fixed_h2=fixed_h2)
        self.T_breaks.append(previous_gens + generations)
        if not trait_updates:
            self.update_traits(fixed_h2=fixed_h2)
        # saves metrics to object
        self.ps = ps
        self._update_metric_history()

    def _pair_mates(self, AM_r: float = 0, AM_trait: Union[str, np.ndarray] = None,
                    AM_type: str = 'phenotypic') -> np.ndarray:
        '''
        Pairs individuals up monogamously to mate and generate offspring. Population size must be multiple of 2. Allows for assortative mating (AM) if specified.
        Parameters:
            AM_r (float): Desired correlation between AM trait values of spouses. Default is 0 (no assortative mating).
            AM_trait (str or 1D array): If a string, the name of the trait to use for assortative mating (as stored in the object). If an array, the trait values to use for assortative mating. If None, no assortative mating is performed. Default is None.
            AM_type (str): Type of assortative mating to perform. If 'phenotypic', uses the trait values directly. If 'genetic', uses the genetic values (only for object's stored traits). Default is 'phenotypic'.

        Returns:
            tuple ((iM, iP)):
            Where:
            - iMs (1D array): Array of length N/2 containing indices of the mothers.
            - iPs (1D array): Array of length N/2 containing indices of the fathers.
        '''
        # checks for population size
        if self.N % 2 != 0:
            raise Exception('Population size must be multiple of 2.')
        N2 = self.N // 2

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

        # randomly splits up population into maternal (M) and paternal (P) halves 
        # also shuffles their order
        iMs = np.random.choice(self.N, N2, replace=False)
        iPs = np.setdiff1d(np.arange(self.N), iMs)

        # computes mate value
        mate_values = AM_r * AM_values + np.random.normal(scale = np.sqrt(1-AM_r**2), size=self.N)
        # sorts mothers and fathers by mate value
        iMs = iMs[np.argsort(AM_values[iMs])]
        iPs = iPs[np.argsort(mate_values[iPs])]    

        return (iMs, iPs)

    def generate_offspring(self, s: Union[float, np.ndarray] = 0,
                           mu: Union[float, np.ndarray] = 0,
                           **kwargs) -> np.ndarray:
        '''
        Pairs up mates and generates offspring for parents' haplotypes. Only works for diploids. Each pair always has two offspring. Recombination rates are extracted from object attributes.
        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float or 1D array): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
            **kwargs: All other arguments (related to assortative mating) are passed to the `_pair_mates` method. See that method for details.
        Returns:
            H (3D array): N*M*P array of offspring haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
        '''
        # checks ploidy
        if self.P != 2:
            raise Exception('Offspring generation only works for diploids.')

        # Assortative Mating ####        
        # pairs up mates
        iMs, iPs = self._pair_mates()

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

        H = np.empty((N_offspring, self.M, self.P), dtype=int)
        for i in np.arange(N_offspring):
            iM = parents[i,0]
            iP = parents[i,1]
            # extract allele from correct haplotype of each parent
            haploM = self.H[iM, np.arange(self.M), haplo_ks[i, :, 0]]
            haploP = self.H[iP, np.arange(self.M), haplo_ks[i, :, 1]]
            haplos = np.stack((haploM, haploP), axis = 1)
            # shuffles haplotypes around
            haplos = haplos[:,np.random.choice(self.P, size=self.P, replace=False)]
            H[i,:,:] = haplos

        # Mutations ####
        if isinstance(mu, (float, int)):
            mu = np.full(self.M, mu)
        mutations = np.random.binomial(n=1, p=mu[None,:,None], size = (N_offspring, self.M, self.P))
        # Apply mutations by flipping alleles (0 to 1 or 1 to 0) based on the mutation matrix
        H = (H + mutations) % 2

        return H

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
        ps = self.ps
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
            ps_mean, ps_quantile = self.summarize_ps(ps, quantiles)
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

    def __init__(self, G: np.ndarray, M_causal: int = None,
                 var_G: float = 1.0, var_Eps: float = 0.0):
        '''
        Initializes and generates trait based on variance components.
        Parameters:
            G (2D array): N*M NON-standardized genotype matrix.
            M_causal (int): Number of causal variants (variants with non-zero effect sizes). Default is all variants.
            var_G (float): Total expected variance contributed by per-standardized-allele genetic effects. Default is 1.0.
            var_Eps (float): Total expected variance contributed by random noise.
        '''
        # generates causal effects
        effects, _ = stat.generate_causal_effects(G.shape[1], M_causal, var_G)
        self._initialize_effects(G, effects, var_Eps)
        # computes trait 
        self.generate_trait(G)

    @classmethod
    def from_effects(cls, G: np.ndarray, effects: np.ndarray, var_Eps: float = 0.0, per_allele: bool = False) -> 'Trait':
        '''
        Initializes a trait from a given genotype matrix and genetic effects array.
        Parameters:
            G (2D array): N*M NON-standardized genotype matrix.
            effects (1D array): M-length array of STANDARDIZED effects. Set non-causal effects to 0.
            per_allele (bool): Whether the effects are per-allele. Default is False.
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
        trait.generate_trait(G, fixed_h2=False)
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

    def generate_trait(self, G: np.ndarray, fixed_h2: bool = False):
        '''
        Generates/updates trait using stored genetic effects and recomputing other components. Automatically updates object's attributes, instead of returning trait values.
        Parameters:
            G (2D array): N*M NON-standardized genotype matrix.
            fixed_h2 (bool): Whether the variance of the noise component should be updated to maintain the heritability. Genetic component must be non-zero. Default is False.
        '''
        N = G.shape[0]
        # makes empty dictionary for trait components
        self.y_ = {}
        # genetic component (using per-allele effects)
        self.y_['G'] = stat.compute_genetic_value(G, self.effects_per_allele)

        var_Eps = self.var['Eps']
        y_nonEps = self.y_['G']
        # recomputes non-noise component to get needed var_Eps to maintain heritability
        if fixed_h2:
            var_Eps = (y_nonEps.var() / self.h2) - y_nonEps.var()

        # random noise component
        self.y_['Eps'] = stat.generate_noise_value(N,var_Eps)

        # computes actual trait as additive of individual components
        self.y = y_nonEps + self.y_['Eps']

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
        # pulls per-allele effects from first population's Trait object   
        effects_per_allele = traits[0].effects_per_allele
        # creates placeholder Trait object (attributes will largely be overwritten)
        trait_new = cls.from_effects(G, effects=effects_per_allele, per_allele=True)
        # concatenates trait components
        trait_new.y = np.concatenate([trait.y for trait in traits])
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
        self.pops = pops
        # initializes basic attributes
        self.era = 0 # starts at 1
        # creates active vector
        self.active = [True] * len(pops)  # active populations as boolean
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