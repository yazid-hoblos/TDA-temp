import pandas as pd
import numpy as np
import networkx as nwx
from ripser import ripser
import persim
from persim import plot_diagrams
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
from scipy.sparse import load_npz
import gudhi as gd
from igraph import Graph
from IPython.display import SVG, display
import matplotlib.pyplot as plt

ppi_df = pd.read_table("../Human_PPI_Network.txt", header=None)
ppi_df.columns = ["ProteinA", "ProteinB", "SemSim"]

complexes_list = []
with open("../CORUM_Human_Complexes.txt") as complexes:
  for line in complexes:
    line = line.strip()
    temp_list = list(line.split("\t"))
    complexes_list.append(temp_list)
    
## We should first explore the individual proteins first
## Get unique proteins from the Complexes file
## Check common proteins
complexes_dict = {}
complexes_single_proteins = []
for idx, cmplx in enumerate(complexes_list):
  for protein in cmplx:
    if protein not in complexes_single_proteins:
      complexes_single_proteins.append(protein)

  complexes_dict[idx] = cmplx

print('There are %d individual proteins in the complexes' % len(complexes_single_proteins))

complexes_proteins_df = ppi_df[(ppi_df['ProteinA'].isin(complexes_single_proteins)) & (ppi_df['ProteinB'].isin(complexes_single_proteins))]
complexes_proteins_idx = {protein: idx for idx, protein in enumerate(complexes_single_proteins)}
c_proteins = len(complexes_single_proteins)

## we should have a dictionary of proteins and all the complexes they belong to as a sort of label for each node
protein_cmplx_aff = {protein : [] for protein in complexes_single_proteins}
 
for idx, cmplx in complexes_dict.items():
    for protein in cmplx:
        protein_cmplx_aff[protein].append(idx)
        
for protein, cmplxs in protein_cmplx_aff.items():
    print(f'Protein {protein} belongs to complex(es): {cmplxs}')


## Visualization
    
cmplx_prot_net = nwx.from_pandas_edgelist(
        complexes_proteins_df,
        source='ProteinA',
        target='ProteinB',
        edge_attr='SemSim'
    )

## The following graph shows us there are a couple of outliers (nodes connected exclusively to eachother)
scaled_degree = [d[1]*100 for d in nwx.degree(cmplx_prot_net)]
nwx.draw_spring(cmplx_prot_net,
        # Weights Based on Column
        #width=list(nwx.get_edge_attributes(cmplx_prot_net, 'SemSim').values()),
        # Node size based on degree
        node_size=10,
        # Colour Based on Degree
        node_color=scaled_degree,
        # Set color map to determine colours
        cmap='rainbow',
        with_labels=False)

# plt.show()

## Let's check out the single-degree nodes and try to identify which protein-pairs form the outliers
one_degree_nodes = []
for degree in cmplx_prot_net.degree():
    if degree[1] == 1:
        one_degree_nodes.append(degree[0])
        
outlier_nodes = []
for edge in cmplx_prot_net.edges(data=True):
    if edge[0] in one_degree_nodes:
        if edge[1] in one_degree_nodes:
            print(edge)
            outlier_nodes.append(edge[0])
            
## Drop data row with those specific nodes and re-define subset
ppi_df[(ppi_df['ProteinA'].isin(outlier_nodes))].head()
complexes_proteins_df = ppi_df[(ppi_df['ProteinA'].isin(complexes_single_proteins)) & (ppi_df['ProteinB'].isin(complexes_single_proteins))]

cmplx_prot_net = nwx.from_pandas_edgelist(
        complexes_proteins_df,
        source='ProteinA',
        target='ProteinB',
        edge_attr='SemSim'
    )

## The following graph shows us there are a couple of outliers (nodes connected exclusively to eachother)
scaled_degree = [d[1]*100 for d in nwx.degree(cmplx_prot_net)]
nwx.draw_spring(cmplx_prot_net,
        # Weights Based on Column
        #width=list(nwx.get_edge_attributes(cmplx_prot_net, 'SemSim').values()),
        # Node size based on degree
        node_size=10,
        # Colour Based on Degree
        node_color=scaled_degree,
        # Set color map to determine colours
        cmap='rainbow',
        with_labels=False)

## Node Degree Viualization
## The following is just to check the distribution of the node degrees. As it seems like there are highly central nodes
degree_nodes = {}
for p, d in cmplx_prot_net.degree():
  degree_nodes[p] = d

## This gives that there are nodes with degree > 900
sorted_node_degrees = dict(sorted(degree_nodes.items(), key=lambda item: item[1],  reverse=True))


## Let's visualize the distribution
# viz_degree = {degree: 0 for degree in degree_nodes.values()}
# for degree in degree_nodes.values():
#   viz_degree[degree]+=1
# degree_count_pairs = sorted(viz_degree.items())
# x, y = zip(*degree_count_pairs) # unpack a list of pairs into two tuples
# plt.plot(x, y)
# plt.xlabel('Node Degree')
# plt.ylabel('Protein Count')
# plt.show()

## Creating adjacency matrix

complexes_adj_mat = np.matrix(np.zeros((c_proteins, c_proteins)))

for idx, row in complexes_proteins_df.iterrows():
  protein_A = complexes_proteins_idx[row['ProteinA']]
  protein_B = complexes_proteins_idx[row['ProteinB']]
  sem_sim_score = row['SemSim']
  complexes_adj_mat[protein_A, protein_B] = sem_sim_score
  complexes_adj_mat[protein_B, protein_A] = sem_sim_score
  
np.fill_diagonal(complexes_adj_mat, 1)
complexes_adj_mat

## Persistence filtration

## Define a function to plot barcode diagrams
def plot_barcode(diag, dim, **kwargs):
    #dim = 0
    diag_dim = diag[dim]
    birth = diag_dim[:, 0]; death = diag_dim[:, 1]
    finite_bars = death[death != np.inf]
    if len(finite_bars) > 0:
        inf_end = 2 * max(finite_bars)
    else:
        inf_end = 2
    death[death == np.inf] = inf_end
    plt.figure(figsize=kwargs.get('figsize', (10, 5)))
    for i, (b, d) in enumerate(zip(birth, death)):
        if d == inf_end:
            plt.plot([b, d], [i, i], color='k', lw=kwargs.get('linewidth', 2))
        else:
            plt.plot([b, d], [i, i], color=kwargs.get('color', 'b'), lw=kwargs.get('linewidth', 2))
    plt.title(kwargs.get('title', 'Persistence Barcode'))
    plt.xlabel(kwargs.get('xlabel', 'Filtration Value'))
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    
cmplx_adj_mat = np.asarray(complexes_adj_mat)
cmplx_dist_mat = 1 - cmplx_adj_mat ## Using the adjacency matrix as a correlation matrix
cmplx_dist_mat

## TESTING
rips_dist_mat_diags = ripser(cmplx_dist_mat, distance_matrix=True)['dgms']
'''
plot_diagrams(rips_dist_mat_diags)

plot_diagrams(rips_dist_mat_diags, plot_only=[0])
plot_diagrams(rips_dist_mat_diags, plot_only=[1])
plot_diagrams(rips_dist_mat_diags, plot_only=[2])
plot_diagrams(rips_dist_mat_diags, plot_only=[3])


## Generate persistence diagrams

tmp_cmplx = complexes_dict[171]
len(tmp_cmplx)

def extract_cmplx_df(cmplx):
    cmplx_source_df = ppi_df[ppi_df['ProteinA'].isin(cmplx)]
    cmplx_target_df = ppi_df[ppi_df['ProteinB'].isin(cmplx)]
    cmplx_df = pd.concat([cmplx_source_df, cmplx_target_df])
    cmplx_df.drop_duplicates()
    return cmplx_df

tmp_df = extract_cmplx_df(tmp_cmplx)

def create_subgraph(cmplx_df):
    tmp_G = nwx.from_pandas_edgelist(
        cmplx_df,
        source='ProteinA',
        target='ProteinB',
        edge_attr='SemSim'
    )
    return tmp_G

tmp_G = create_subgraph(tmp_df)

nwx.draw_spring(tmp_G)

for idx, row in tmp_df.iterrows():
    if row['SemSim'] == 0.0:
        tmp_df.drop(index=idx)
        
tmp_G = create_subgraph(tmp_df)

def get_cmplx_dist_mat(cmplx_graph):
    cmplx_adj_mat = nwx.adjacency_matrix(cmplx_graph).toarray()
    np.fill_diagonal(cmplx_adj_mat, 1)
    cmplx_dist_mat = 1 - cmplx_adj_mat
    return cmplx_dist_mat

tmp_dist_mat = get_cmplx_dist_mat(tmp_G)

def generate_PD(cmplx_corr_dist_mat, cmplx_idx):
    cmplx_PD = ripser(cmplx_corr_dist_mat, distance_matrix = True)['dgms']
    diag_title = f'Persistence Diagram for Complex {cmplx_idx}'
    plot_diagrams(cmplx_PD, title=diag_title, legend= False)
    png_title = f'cmplx_pers_diags/{diag_title}.png'
    plt.savefig(png_title)
    cmplx_PD = None
    plt.close()
    
generate_PD(tmp_dist_mat, 171)

for idx, cmplx in complexes_dict.items():
     if len(cmplx) >= 3:
          cmplx_df = extract_cmplx_df(cmplx)
          tmp_cmplx_G = create_subgraph(cmplx_df)
          cmplx_dist_mat = get_cmplx_dist_mat(tmp_cmplx_G)
          generate_PD(cmplx_dist_mat, idx)
          
## Gudhi

protein_net = nwx.from_pandas_edgelist(
    ppi_df,
    source='ProteinA',
    target='ProteinB',
    edge_attr='SemSim'
)

ppi_proteins = np.unique(ppi_df[['ProteinA', 'ProteinB']])

## Let's store the proteins in an indexed dictionary so we could track them back when building the simplex tree
proteins_dict = {protein: idx for idx, protein in enumerate(ppi_proteins)}

# Given protein-protein interaction network we can start by creating a simplex tree that includes all 0-simplices (nodes)

# Construct a simplex tree from the network
simplex_tree = gd.SimplexTree()

for edge in protein_net.edges(data=True):
    node1, node2, weight = edge
    ## Get protein index from dict to map it back and feed it into the simplex tree
    node1_idx = proteins_dict[node1]
    node2_idx = proteins_dict[node2]
    simplex_tree.insert([node1_idx, node2_idx], filtration=weight['SemSim'])


# Compute persistence diagrams
## NOTE: min_persistence is set to -1 to view all the simplex values (Include all 0-simplices)

persistence = simplex_tree.persistence(min_persistence=0, persistence_dim_max=True)

# Generate persistence diagrams
diagrams = gd.plot_persistence_diagram(persistence)
barcode = gd.plot_persistence_barcode(persistence)
density = gd.plot_persistence_density(persistence)

for smplx in persistence:
    if smplx[0] == 0 and smplx[1][1] != np.inf:
        print(smplx)
        
st_filt_gen = simplex_tree.get_filtration()

for smplx in st_filt_gen:
    print(smplx)
    
simplex_tree.persistence_intervals_in_dimension(0)


## Gudhi RipsComplex construction

## Here we use the Gudhi library to build the Rips complex and apply the homology.
## We can build the Rips simplicial complex by using a distance matrix. So I just plugged in the correlation distance matrix.
rips_complex = gd.RipsComplex(distance_matrix=pn_distance_mat, max_edge_length=1.0)


## We now build a simplex tree to store the simplices
rips_simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)


## Check how the complex looks like
result_str = 'Rips complex is of dimension ' + repr(rips_simplex_tree.dimension()) + ' - ' + \
    repr(rips_simplex_tree.num_simplices()) + ' simplices - ' + \
    repr(rips_simplex_tree.num_vertices()) + ' vertices.'

print(result_str)

persistence = rips_simplex_tree.persistence(min_persistence=-1, persistence_dim_max=True)

# Generate persistence diagrams
diagrams = gd.plot_persistence_diagram(persistence, max_intervals=4000000)
'''