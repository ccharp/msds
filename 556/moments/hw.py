import numpy as np
from scipy import stats as st

def show_stats(dist_str, sample):
    sample.sort()
    print("The simulated median of " + dist_str + " is " + str(sample[sample.size // 2]))
    print("  with mode of: " + str(st.mode(sample).mode))

# 1a.
show_stats("Unif(0, 2)", np.random.uniform(0, 2, 1000))

# 2a.
show_stats("Expo(2)", np.random.exponential([0.5]*1000))

# 3a.
show_stats("DUnif(1, 10", np.random.randint(1, 10, 1000))

# Note: our mode simulation for both continuous and discrete uniforms is incomplete/limited. We'd nee
#  find the mode across multple samples drawn from the same distribution and then show that the entire
#  uniform line is equally likely. 