import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union

class Population:
    '''
    Class for a population to simulate. Contains genotype information. Contains methods to simulate change in population over time.
    '''

    ########################
    #### Initialization ####
    ########################

    def __init__(self, N: int, M: int, P: int = 2,
                 p_init: Union[float, np.ndarray] = None,
                 seed: int = None):
        '''
        Initializes a population, simulating initial genotypes.

        Parameters:
            N (int): {opulation size of individuals (not haplotypes).
            M (int): Total number of variants in genome.
            P (int): ploidy of genotpes. Default is 2 (diploid).
            p_init (float or array): Initial allele frequency of variants. If only a single value is provided, it is treated as the initial allele frequency for all variants. Otherwise, must be an array of length M. Default is uniform distribution between 0.05 and 0.95.
            seed (int): Initial seed to use when simulating genotypes (and allele frequencies if necessary).
        '''
        # defines properties of instance
        self.N = N
        self.M = M
        self.P = P
        self.T_breaks = [0]
        # sets seed if specified
        if seed is not None:
            seed = np.random.seed(seed)
        # draws initial allele frequencies from uniform distributio between 0.05 and 0.95 if not specified
        if p_init is None:
            p_init = self._draw_p_init(method = 'uniform', params = (0.05, 0.95))
        elif type(p_init) == float or type(p_init) == int:
            # if only single value is given, all variants have same initial frequency
            p_init = np.full(self.M, p_init)

        # generates initial genotypes and records allele frequencies
        H = self._generate_unrelated_haplotypes(p_init)
        self._update_obj(H=H)
        self.ps = np.expand_dims(self.p, axis=0)

        # generates initial kinship matrix
        self.K = np.diag(np.full(self.N, 1))

    def _draw_p_init(self, method: str, params: list) -> np.ndarray:
        '''
        Returns array of initial allele frequencies to simulate genotypes from.

        Parameters:
            method (str): Method of randomly drawing allele frequencies. Options:
                uniform: 'Draws from uniform distribution with lower and upper bounds specified by `params`.'
            params (list): Method-specific parameter values.
        Returns:
            p_init (1D array): Array of length M containing allele frequencies.
        '''
        # uniform sampling is default
        if method == 'uniform':
            p_init = np.random.uniform(params[0], params[1], self.M)
            return p_init
        else:
            return np.full(self.M, np.nan)
    
    ###################################
    #### Setting object attributes ####
    ###################################

    def _update_obj(self, H: np.ndarray = None, K: np.ndarray = None):
        '''
        Update the population object's attributes.

        Parameters:
            H (3D array): Haplotype array.
            K (2D array): Kinship array.
        '''

        if H is not None:
            self.H = H
            self.G = self._get_G(self.H)
            self.p = self.get_freq(self.G)
        if K is not None:
            self.K = K

    @staticmethod
    def _get_G(H) -> np.ndarray:
        '''
        Collapses haplotypes into genotypes.

        Parameters:
            H (3D array): N*M*P array of alleles
        Returns:
            G (2D array): N*M array of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
        '''
        G = H.sum(axis=2)
        return G

    def get_freq(self, G) -> np.ndarray:
        '''
        Return array of allele frequencies for current genotypes.

        Parameters:
            G (2D array): N*M array of genotypes. First dimension is individuals, second dimension is variants. Each element is an integer ranging from 0 to P (the ploidy).
        Returns:
            p (1D array): Array of allele frequencies.
        '''
        p = G.mean(axis=0) / self.P
        return p
    
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

    ####################################
    #### Simulating forward in time ####
    ####################################

    def _generate_unrelated_haplotypes(self, p: np.ndarray) -> np.ndarray:
        '''
        Generates 3-dimensional matrix of population haplotypes.

        Parameters:
            p (1D array): Array of allele frequencies to draw alleles from.
        Returns:
            H (3D array): N*M*P array of alleles. First dimension is individuals, second dimension is variants, third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
        '''
        p = p.reshape(1, self.M, 1)
        H = np.random.binomial( 1, p = p, size = (self.N, self.M, self.P))
        return H
        
    def next_generation(self, s: Union[float, np.ndarray] = 0, mu: float = 0):
        '''
        Simulates new generation. Doesn't simulate offspring directly, meaning that future offspring have haplotypes drawn randomly from allele frequencies. Automatically updates object.

        Parameters:
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
        '''
        # assigns same selection coefficient/mutation rate to all variants if only single value specified
        if type(s) == float:
            s = np.full(self.M, s)
        if type(mu) == float:
            mu = np.full(self.M, mu)
        # analytical allele frequency from which next generations' alleles are drawn from
        p = self.p
        # effect of selection
        p = p * (1 + s) / (1 + p*s)
        # effect of mutation (in gametes that lead to new generation, i.e. post-selection)
        p = p*(1-mu) + (1-p)*mu
        # effect of genetic drift
        H = self._generate_unrelated_haplotypes(p)
        self._update_obj(H=H)

    def simulate_generations(self, generations: int, record_history: bool = True,
                             mu: float = 0, s: Union[float, np.ndarray] = 0):
        '''
        Simulates specified number of generations beyond current generation. Doesn't simulate offspring directly, meaning that future offspring have haplotypes drawn randomly from allele frequencies. Automatically updates object.

        Parameters:
            generations (int): Number of generations to simulate (beyond the current generation).
            record_history (bool): Determines if allele frequencies at each generation are saved to a matrix belonging to the object. Default is True.
            s (float or 1D array): Selection coefficient, such that an individual with the alternate allele has a (1+s) relative fitness compared to the reference allele. Occurs before mutation. If only a single value is provided, it is treated as the selection coefficient for all variants. Otherwise, must be an array of length M. Default is 0 (no selection).
            mu (float): Mutation rate, such that the probability of any individual allele flipping to its alternate in the next generation is given by mu. Occurs after selection (i.e. mutation occurs in germline of current generation). Default is 0 (no mutations).
        '''
        # keeps track of allele frequencies over generations if specified
        if record_history:
            # checks if previous generations have already been generated or not
            previous_gens = self.ps.shape[0]
            ps = np.full( (previous_gens + generations, self.M), np.nan)
            ps[0:previous_gens,] = self.ps
        
        # loops through each generation
        for t in range(generations):
            self.next_generation(mu=mu, s=s)
            # records allele frequency
            if record_history:
                ps[previous_gens + t,] = self.p
        self.T_breaks.append(previous_gens + generations)
        # saves allele freqs. at each generation (keeps old generations)
        if record_history:
            self.ps = ps


    def _pair_mates(self) -> np.ndarray:
        '''
        Pairs individuals up monogamously to mate and generate offspring. Population size must be multiple of 2. In the future will allow for assortative mating (phenotypic and genetic). 

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
        # randomly splits up population into maternal (M) and paternal (P) halves 
        # also shuffles their order
        iMs = np.random.choice(self.N, N2, replace=False)
        iPs = np.setdiff1d(np.arange(self.N), iMs)

        return (iMs, iPs)

    def generate_offspring(self, replace: bool = True):
        '''
        Pairs up mates and generates offspring for parents' haplotypes. Only works for diploids. Each pair always has two offspring.

        Parameters:
            replace (bool): Whether the offspring replace the current generation. Default is True.
        
        Returns:
            H (3D array): N*M*P array of offspring haplotypes. First dimension is individuals, second dimension is variants, and third dimension is haplotype number (related to ploidy). Each element is either a 0 or a 1.
        '''
        # checks ploidy
        if self.P != 2:
            raise Exception('Offspring generation only works for diploids.')
        # pairs up mates
        iMs, iPs = self._pair_mates()

        # determines population size of next generation (currently maintains population size)
        N_pairs = self.N // 2
        N_offspring_per_pair = 2 # to maintain population size
        N_offspring = N_pairs * N_offspring_per_pair
        
        # randomly chooses haplotype to keep from each parent
        hMs = np.random.binomial(1, 0.5, size = (N_pairs, N_offspring_per_pair))
        hPs = np.random.binomial(1, 0.5, size = (N_pairs, N_offspring_per_pair))

        # initializes arrays
        H = np.empty((N_offspring, self.M, self.P), dtype=int)
        K_MP = self.K.copy() # parental kinship matrix
        K = np.diag(np.full(N_offspring, 0.5)) # offspring kinship matrix
        # loops through each mate pair
        for i in range(N_pairs):
            iM = iMs[i]
            iP = iPs[i]
            # indices of offspring
            iOs = (N_offspring_per_pair*i)+np.arange(N_offspring_per_pair)
            # looks through each offspring of pair
            for j in range(len(iOs)):
                iO = iOs[j]
                haploM = self.H[iM,:,hMs[i,j]]
                haploP = self.H[iP,:,hPs[i,j]]
                haplos = np.stack((haploM, haploP), axis = 1)
                # shuffles haplotypes around
                haplos = haplos[:,np.random.choice(self.P, size=self.P, replace=False)]
                H[iO,:,:] = haplos

                # updates lower triangle of kinship matrix
                for jj in range(j + 1, len(iOs)):
                    K[iOs[j], iOs[jj]] = (1 + K_MP[iM, iP]) / 2

        # fils out upper triangle of kinship matrix:
        K += K.T

        if replace:
            self._update_obj(H=H, K=K)
        else:
            return (H, K)

    #######################
    #### Visualization ####
    #######################

    def plot_freq_over_time(self, ps: np.ndarray = None, j_keep: tuple = None,
                            legend=False, last_generations: int = None,
                            summarize: bool = False, quantiles: tuple = (0.25, 0.5, 0.75)):
        '''
        Plots variant allele frequencies over time.

        Parameters:
            ps (2D array): T*M matrix (where T is number of generations) containing allele frequencies over time. Defaults to object's allele frequency history.
            j_keep (tuple): Variant indices to include when plotting. Defaults to all variants.
            legend (bool): Whether to include a legend in the plot for each line. Default is False.
            last_generations (int): Number specifying the number of most recent generations to plot. Defaults to all generations since beginning.
            summarize (bool): If true, instead of plotting individual variant trajectories, it plots the mean and specified quantiles of allele frequencies across variants at each generation. Default is False.
            quantiles (tuple): List of quantiles (e.g. 0.99) of allele frequencies across variants at each generation to plot. `summarize` must be set to True. Default is median, lower quartile, and upper quartile.
        '''
        # plots all variants if not specified
        if j_keep is None:
            j_keep = tuple( range(self.M) )
        # uses population's allele frequency history if not specified
        if ps is None:
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
        
        # plotting
        plt.figure(figsize=(8, 5))
        # allele frequency lines
        if not summarize:
            for j in range(ps.shape[1]):
                plt.plot(ts, ps[:, j], label=f'Variant {j_keep[j]}')
        else:
            plt.plot(ts, ps_mean, color='deepskyblue', label = 'Mean', ls='--')
            for j in range(len(quantiles)):
                plt.plot(ts, ps_quantile[j,:], label=f'{quantiles[j]*100}% percentile', color = 'lightskyblue', ls=':')

        # vertical lines denoting simulation batches
        for t in self.T_breaks:
            plt.axvline(t, ls='--', color='black')
        # labels
        plt.xlabel('Generation')
        plt.ylabel('Allele Frequency')
        plt.title('Allele Frequency Trajectories Over Time')
        plt.xlim(ts.min(), ts.max())
        plt.ylim(0, 1)
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()