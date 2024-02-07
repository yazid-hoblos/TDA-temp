import numpy as np
import ripser
import pandas as pd
import scipy.sparse

ppi_df = pd.read_table("biogrid_corr_dist_mat.csv", header=None, delimiter=',')
print('loaded')
ppi_npz = scipy.sparse.csr_matrix(ppi_df.values)
print('converted')
dist_mat_diags_ripser = ripser.ripser(ppi_npz, distance_matrix=True, maxdim=3)['dgms']