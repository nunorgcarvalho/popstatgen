'''
This file contains functions related to statistical genetics.
The `popsim.py` file contains classes that contain wrapper methods that call these functions, providing the class object's attributes as arguments.
Alternatively, these functions can be called directly with the appropriate arguments.
The functions here contain the documentation for the arguments and return values, which is not repeated in the class methods for the most part.
'''

# imports
import numpy as np

##########################
#### Trait generation ####
##########################

def generate_causal_effects(M: int, M_causal: int = None, var_G: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates variant effect sizes for some trait. Default interpretation is per-standardized-allele effect sizes. Non-causal variant effects are set to 0. Causal effects are drawn from normal distribution with mu=0 and sd=`var_G`/`M_causal`.
    Parameters:
        M (int): Total number of variants (causal and non-causal).
        M_causal (int): Number of causal variants (variants with non-zero effect sizes). Default is all variants.
        var_G (float): Total expected variance contributed by per-standardized-allele genetic effects. Default is 1.0.
    Returns:
        tuple ((causal_effects, j_causal)):
        Where:
        - causal_effects (1D array): Array of length M containing the effect sizes of all variants.
        - j_causal (1D array): Array of length M_causal containing the variant indices of causal variants.
    '''
    if M_causal is None:
        M_causal = M
    causal_effects = np.zeros(M)
    j_causal = np.random.choice(M, M_causal, replace=False) 
    causal_effects[j_causal] = np.random.normal(0, np.sqrt(var_G/M_causal), M_causal)
    return (causal_effects, j_causal)

def compute_genetic_value(G: np.ndarray, effects: np.ndarray) -> np.ndarray:
    '''
    Computes the genetic value/score given a (standardized) genotype matrix and (per-standardized-allele) effect sizes.
    Parameters:
        G (2D array): N*M NON-standardized genotype matrix.
        effects (1D array): Array of length M containing causal and non-causal genetic effects.
    Returns:
        y_G (1D array): Array of length N containing genetic component values.
    '''
    # dot product
    y_G = G @ effects
    return y_G

def generate_noise_value(N: int, var_Eps: float = 0.0) -> np.ndarray:
    '''
    Generates noise component of trait drawn randomly from Normal distribution.
    Parameters:
        N (int): Number of individuals to generate noise component for.
        var_Eps (float): Variance of the noise component. Default is 0.
    Returns:
        y_Eps (1D array): Array of length N containing noise component values.
    '''
    y_Eps = np.random.normal(loc=0, scale=np.sqrt(var_Eps), size = N)
    return y_Eps

def get_standardized_effects(effects: np.ndarray, G: np.ndarray, std2allelic: bool = True) -> np.ndarray:
        '''
        Converts between per-allele and per-standardized-allele effects.
        Parameters:
            effects (1D array): M-length array of effects. Can be per-allele or per-standardized-allele.
            G (2D array): N*M NON-standardized genotype matrix.
            std2allelic (bool): If True (default), converts from per-standardized-allele to per-allele effects. If False, converts from per-allele to per-standardized-allele effects.
        '''
        # gets observed standard deviation of the genotype matrix
        G_std = G.std(axis=0)
        # if an allele is monomorphic, it uses the standard deviation one would get for a binomial variable with p = 1 / (N * P)
        P = G.max() # estimates ploidy to be the maximum value in G, which may not always be true
        G_std[G_std == 0] = 1 / (G.shape[0] * P)

        if std2allelic:
            effects_output = effects / G_std # per-allele effects
        else:
            effects_output = effects * G_std # per-standardized-allele effects
        return effects_output