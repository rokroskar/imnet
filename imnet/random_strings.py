import random


def set_seed(seed):
    random.seed(seed)


def generate_random_sequences(
    N, minimum_length=4, maximum_length=20, seed=1, letters="ACDEFGHIKLMNPQRSTVWY"
):
    """
    Generate a list of random strings

    Parameters
    ---------- 

    N : int 
        number of strings
    minimum_length : int, optional
        minimum number of characters
    maximum_length : int, optional
        maximum number of characters
    seed : int, optional  
        random seed 
    letters : list, optional
        a list of letters to use for generating random sequences
    """
    set_seed(seed)

    string_list = []

    for i in range(N):
        random_integer = random.randrange(minimum_length, maximum_length)
        s = "".join(random.choice(letters) for _ in range(random_integer))
        string_list.append(s)

    return string_list
