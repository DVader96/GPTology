from scipy.stats import pearsonr
import numpy as np
from embed_post_pro import project_out

l22 = np.random.randn(1709)
l2 = l22 + 0.1*np.random.randn(1709)

signal = 5*l22 + 2

out = project_out(l2, l22)

print(pearsonr(out, signal))
