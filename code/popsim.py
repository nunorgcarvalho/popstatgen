import numpy as np

class Population:
    # Type hints
    N: int # population size of individuals (not haplotypes)
    M: int # total number of variants in genome
    P: int # ploidy, default = 2
    H: np.ndarray # genotypes, phased
    G: np.ndarray # genotypes, unphased
    p: np.ndarray # current allele frequencies

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

        self.H = self._generate_unrelated_haplotypes(p_init)
        self.G = self._get_G()
    

    def _draw_p_init(self, method: str, params: list) -> np.ndarray:
        """
        returns array of initial allele frequencies to simulate genotypes from
        """
        # uniform sampling is default
        if method == 'uniform':
            p_init = np.random.uniform(params[0], params[1], self.M)
            return p_init
        else:
            return np.zeros(self.M)
    
    def _generate_unrelated_haplotypes(self, p_init: np.ndarray) -> np.ndarray:
        """
        generates haplotype 3-dimensional matrix
        """
        p_broadcast = p_init.reshape(1, self.M, 1)
        H = np.random.binomial( 1, p = p_broadcast, size = (self.N, self.M, self.P))
        return H
    
    def _get_G(self) -> np.ndarray:
        """
        collapses haplotypes into genotypes
        """
        G = self.H.sum(axis=2)
        return G

    def get_freq(self) -> np.ndarray:
        """
        Return array of allele frequencies for current genotypes
        """
        p = self.G.mean(axis=0) / self.P
        return p
        