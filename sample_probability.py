import numpy as np


def sample_probability(bool_array):
    """
    Compute the fraction of True values in a NumPy boolean array.

    Parameters
    ----------
    bool_array : np.ndarray
        A NumPy array of booleans (dtype=bool).

    Returns
    -------
    float
        Number of True entries divided by total number of entries.
        Returns 0.0 if the array is empty.
    """
    bool_array = np.asarray(bool_array, dtype=bool)

    if bool_array.size == 0:
        return 0.0

    return np.mean(bool_array)





