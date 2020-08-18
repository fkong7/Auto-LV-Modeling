import numpy as np

def sample(iterable, n):
    """
    Returns @param n random items from @param iterable.
    """
    if n == 0:
      return []
    reservoir = []
    factor = int(np.ceil(float(n)/float(len(iterable))))
    for i in range(0,factor-1):
      iterable+=iterable
    for t, item in enumerate(iterable):
        if t < n:
            reservoir.append(item)
        else:
            m = np.random.randint(0,t)
            if m < n:
                reservoir[m] = item
    return reservoir

def bootstrapping(fn_names):
    index = np.array(range(len(fn_names)))
    new_fn_names = [None]*len(fn_names)
    for i in range(len(fn_names)):
        new_fn_names[i] = fn_names[int(np.random.choice(index,replace=True))]

    return new_fn_names

