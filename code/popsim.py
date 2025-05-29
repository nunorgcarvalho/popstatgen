import numpy as np
import matplotlib.pyplot as plt

class Population:
    """
    Class for a population to simulate. Contains genotype information. Contains methods to simulate change in population over time.
    """
    # Type hints
    N: int # population size of individuals (not haplotypes)
    M: int # total number of variants in genome
    P: int # ploidy, default = 2

    H: np.ndarray # genotypes, phased
    G: np.ndarray # genotypes, unphased
    p: np.ndarray # current allele frequencies

    ps: np.ndarray # array of allele frequencies over generations

    # initializes instance
    def __init__(self, N: int, M: int,
                 p_init: np.ndarray = None,
                 P: int = 2,
                 seed: int = None):
        # defines properties of instance
        self.N = N
        self.M = M
        self.P = P
        # sets seed if specified
        if seed is not None:
            seed = np.random.seed(seed)
        if p_init is None:
            p_init = self._draw_p_init(method = 'uniform', params = (0.05, 0.95))
        elif type(p_init) == float or type(p_init) == int:
            # if only single value is given, all variants have same initial frequency
            p_init = np.full(self.M, p_init)

        # generates initial genotypes and records allele frequencies
        self.H = self._generate_unrelated_haplotypes(p_init)
        self.G = self._get_G(self.H)
        self.p = self.get_freq(self.G)
        self.ps = np.expand_dims(self.p, axis=0)

    def _draw_p_init(self, method: str, params: list) -> np.ndarray:
        """
        Returns array of initial allele frequencies to simulate genotypes from
        """
        # uniform sampling is default
        if method == 'uniform':
            p_init = np.random.uniform(params[0], params[1], self.M)
            return p_init
        else:
            return np.full(self.M, np.nan)
    
    def _generate_unrelated_haplotypes(self, p_init: np.ndarray) -> np.ndarray:
        """
        Generates haplotype 3-dimensional matrix
        """
        p_broadcast = p_init.reshape(1, self.M, 1)
        H = np.random.binomial( 1, p = p_broadcast, size = (self.N, self.M, self.P))
        return H
    
    @staticmethod
    def _get_G(H) -> np.ndarray:
        """
        Collapses haplotypes into genotypes
        """
        G = H.sum(axis=2)
        return G
    
    def get_freq(self, G) -> np.ndarray:
        """
        Return array of allele frequencies for current genotypes
        """
        p = G.mean(axis=0) / self.P
        return p
    
    def next_generation(self):
        """
        Simulates new generation.
        Doesn't simulate offspring directly.
        """
        self.H = self._generate_unrelated_haplotypes(self.p)
        self.G = self._get_G(self.H)
        self.p = self.get_freq(self.G)

    def simulate_generations(self, generations: int, update_record: bool = True):
        """
        Simulates specified number of generations beyond current
        """
        # keeps track of allele frequencies over generations if specified
        if update_record:
            # checks if previous generations have already been generated or not
            previous_gens = self.ps.shape[0]
            ps = np.full( (previous_gens + generations, self.M), np.nan)
            ps[0:previous_gens,] = self.ps
        
        # loops through each generation
        for t in range(generations):
            self.next_generation()
            # records allele frequency
            if update_record:
                ps[previous_gens + t,] = self.p
        
        # saves allele freqs. at each generation (keeps old generations)
        if update_record:
            self.ps = ps

    def plot_freq_over_time(self, ps: np.ndarray = None, j_keep: list = None, legend=False):
        """
        Returns plot of variant allele frequencies over time
        """
        # plots all variants if not specified
        if j_keep is None:
            j_keep = list( range(self.M) )
        # uses population's allele frequency history if not specified
        if ps is None:
            ps = self.ps
        # subsets to specified variants
        ps = ps[:,j_keep]
        # plots
        T = np.arange(ps.shape[0])
        plt.figure(figsize=(8, 5))
        for j in range(ps.shape[1]):
            plt.plot(T, ps[:, j], label=f'Variant {j_keep[j]}')
        plt.xlabel('Generation')
        plt.ylabel('Allele Frequency')
        plt.title('Allele Frequency Trajectories Over Time')
        plt.ylim(0, 1)
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    def get_fixation_t(self, ps: np.ndarray = None) -> np.ndarray:
        """
        For each variant, finds *first* generation (returned as an index) for which the allele frequency is 0 or 1 (fixation).
        A value of -1 means the variant never got fixed
        """
        # uses population's allele frequency history if not specified
        if ps is None:
            ps = self.ps
        # gets mask of whether frequency is 0 or 1
        ps_mask = np.any((ps == 0, ps == 1), axis=0)
        # finds first instance of True for each variant
        t_fix = np.where(ps_mask.any(axis=0), ps_mask.argmax(axis=0), -1)        
        return t_fix
        