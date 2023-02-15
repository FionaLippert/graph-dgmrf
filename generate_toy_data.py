import torch
import torch_geometric as ptg
import torch.distributions as dists
import numpy as np
import scipy
import networkx as nx
import os
import argparse
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky, analyze, analyze_AAt
import torch_sparse

from layers import chol

import visualization as vis
import constants
import utils

# Always on cpu
MASK_LIMITS = torch.tensor([[0.5, 0.3], [0.9, 0.9]])
MEAN_STD = 1.5 # Std.-dev. for random mean

def densify_adj(A, n):
    orig_A = A
    for _ in range(n):
        A = A + orig_A@A

    A = torch.clamp(A, 0., 1.)
    A.fill_diagonal_(0.)
    return A

def construct_prec(A, alpha1, alpha2, layers):
    node_degrees = torch.sum(A, dim=0)
    # alpha1*Degree on diagonal, alpha2 at neighbor positions
    G = args.alpha2*A + args.alpha1*torch.diag(node_degrees)

    layered_G = G
    for i in range(layers-1):
        layered_G = layered_G@G

    precision_matrix = layered_G.transpose(0,1)@layered_G
    return precision_matrix

def sample_using_prec(precision_matrix, diag_eps, mean):
    num_nodes = precision_matrix.shape[0]
    precision_matrix += diag_eps*torch.eye(num_nodes)

    mean_vec = mean*torch.ones(num_nodes)

    # Sample x
    normal = dists.MultivariateNormal(loc=mean_vec,
            precision_matrix=precision_matrix)
    x = normal.sample().unsqueeze(1)
    return x, normal.covariance_matrix

def sample_gmrf(A, alpha1, alpha2, layers, diag_eps, mean):
    prec = construct_prec(A, alpha1, alpha2, layers)
    x, cov = sample_using_prec(prec, diag_eps, mean)
    return x, prec, cov

parser = argparse.ArgumentParser(description='Generate dataset')

parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--plot", type=int, default=0,
        help="If plots should be made  during generation")

# Graph construction
parser.add_argument("--num_nodes", type=int, default=2000,
        help="Number of nodes in graph")
parser.add_argument("--graph_alg", type=str, default="delaunay",
        help="Algorithm to use for constructing graph")
parser.add_argument("--n_neighbors", type=int, default=5,
        help="Amount of neighbors to include in k-nn graph generation")
parser.add_argument("--noise_std", type=float, default=1e-2,
        help="Std.-dev. of noise for p(y|x)")
parser.add_argument("--random_mask", type=int, default=1,
        help="Use a random mask rather than cut-out areas")
parser.add_argument("--mask_fraction", type=float, default=0.2,
        help="Fraction of points to mask")

# True parameters
parser.add_argument("--diag_eps", type=float, default=1e-6,
    help="Small epsilon added to diagonal to prevent 0 determinant")
parser.add_argument("--alpha1", type=float, default=1.0,
        help="alpha1 parameter, node self-weight")
parser.add_argument("--alpha2", type=float, default=-1.0,
        help="alpha2 parameter, node neighbor-weight")
parser.add_argument("--zero_mean", type=int, default=1,
        help="If distribution should have zero mean (otherwise sample random mean)")

# Perturbations, more complicated models
parser.add_argument("--densify_layers", type=int, default=1,
        help="Number of layers to construct adjacency matrix using")
parser.add_argument("--layers", type=int, default=1,
        help="Number of layers in true model")
parser.add_argument("--sample_mix", type=int, default=0,
        help="Sample from multiple GMRFs and take random linear combination of samples")
parser.add_argument("--prec_mix", type=int, default=0,
        help="Sample random precision matrix with given spatial neighborhood")

# Distance weighting
parser.add_argument("--dist_weight", type=int, default=0,
        help="Use distance weighting of nodes")
parser.add_argument("--dist_weight_eps", type=int, default=1e-2,
        help="Epsilon to add to distances to prevent division by zero")

# Log-det pre-processing
parser.add_argument("--compute_eigvals", type=int, default=1,
        help="If eigenvalues should be computed")
parser.add_argument("--compute_dad_traces", type=int, default=0,
        help="If traces of the DAD-matrix should be estimated")
parser.add_argument("--dad_samples", type=int, default=1000,
        help="Amount of samples to use in DAD trace estimates")
parser.add_argument("--dad_k_max", type=int, default=50,
        help="Maximum k to compute DAD trace for")

# new options for Cholesky factor approach
parser.add_argument("--compute_cholesky", type=int, default=1,
        help="If sparsity pattern of the Cholesky factor should be computed")

args = parser.parse_args()

_ = torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

if args.sample_mix:
    ds_name = "gmrf_mix"
elif args.prec_mix:
    ds_name = "gmrf_prec_mix"
else:
    ds_name = "toy_gmrf"

# Generate graph
print("Generating graph ...")
node_pos = torch.rand(args.num_nodes, 2)

point_data = ptg.data.Data(pos=node_pos)
if args.graph_alg == "delaunay":
    graph_transforms = ptg.transforms.Compose((
        ptg.transforms.Delaunay(),
        ptg.transforms.FaceToEdge(),
    ))
elif args.graph_alg == "knn":
    graph_transforms = ptg.transforms.Compose((
        ptg.transforms.KNNGraph(k=args.n_neighbors, force_undirected=True),
    ))
else:
    assert False, "Unknown graph algorithm"

graph_pos = graph_transforms(point_data)

# compute sparsity pattern of Cholesky factor
if args.compute_cholesky:
    #A = ptg.utils.to_scipy_sparse_matrix(graph_pos.edge_index) #.tocsc()
    laplacian = ptg.utils.get_laplacian(graph_pos.edge_index)
    laplacian = ptg.utils.to_scipy_sparse_matrix(*laplacian).tocsc()
    print(f'number of non-zero entries in laplacian: {laplacian.nnz}')

    eps = 0.00001 # small perturbation to prevent zero entries on diagonal
    Q = laplacian + eps * scipy.sparse.eye(graph_pos.num_nodes, format='csc')

    factor = cholesky(Q, use_long=False)#, mode='simplicial') # use analyse and inplace_cholesky if same graph is used multiple times
    perm = factor.P() # fill-reducing permutation matrix
    L = factor.L() # sparsity pattern of Cholesky factor
    edge_index_L, edge_attr_L = ptg.utils.from_scipy_sparse_matrix(L)
    print(f'number of non-zero entries in L: {L.nnz} ({L.nnz / laplacian.nnz:3.2f} times as much as original graph)')

    factor = cholesky(Q, use_long=False, ordering_method='natural') # no permutation
    Pn = factor.P()  # fill-reducing permutation matrix
    Ln = factor.L()  # sparsity pattern of Cholesky factor
    print(f'number of non-zero entries in Ln: {Ln.nnz} ({Ln.nnz / L.nnz:3.2f} times as much as L)')

    Q_perm = Q[perm[:, np.newaxis], perm[:, np.newaxis]]

    # TODO: apply permutation to all graph attributes (observations, gt_states, edge_index, edge_attr, covariates, ...)
    # from then on, only work with the new ordering
    graph_pos.edge_index, _ = ptg.utils.from_scipy_sparse_matrix(Q_perm)
    graph_pos.pos = graph_pos.pos[perm]
    graph_pos.degree = ptg.utils.degree(graph_pos.edge_index[0], num_nodes=graph_pos.num_nodes)
    graph_pos.node_features = torch.cat([graph_pos.pos, graph_pos.degree.view(-1, 1)], dim=1)

    adj_L = torch_sparse.tensor.from_scipy(L)
    graph_pos.adj_L = adj_L

    # add distances
    graph_pos.edge_attr = torch.norm(graph_pos.pos[graph_pos.edge_index[0]] - graph_pos.pos[graph_pos.edge_index[1]],
                                     p=2, dim=-1).view(-1, 1)


    # test triangular solve
    # TODO: test multiple rhs
    solver = chol.TestSolve()
    x = solver.triangular_solve(adj_L, torch.ones(graph_pos.num_nodes))

    print(x)

    print(f'reconstruction = {L @ x.view(-1, 1)}')

    x = solver.triangular_solve(adj_L.t(), torch.ones(graph_pos.num_nodes), backward=1)

    print(x)

    print(f'reconstruction = {L.transpose() @ x.view(-1, 1)}')






# Construct G, Q
if args.dist_weight:
    utils.dist_weight_graph(graph_pos, args.dist_weight_eps)
    adj_matrix = ptg.utils.to_dense_adj(graph_pos.edge_index,
            edge_attr=graph_pos.edge_attr).squeeze()
else:
    adj_matrix = ptg.utils.to_dense_adj(graph_pos.edge_index)[0]

# Optionally densify sampling adjacency matrix
if args.densify_layers > 1:
    adj_matrix = densify_adj(adj_matrix, (args.densify_layers - 1))

print("Sampling x and y ...")
graph_x = graph_pos.clone()
expl_gmrf = False # If sample is from GMRF on explicit form that we can analyze
if args.sample_mix:
    # Sample from random GMRFs and take a random linear combination of samples
    gmrf_samples = []
    denseness = torch.randint(0,4,(args.sample_mix,))
    means = MEAN_STD*torch.randn(args.sample_mix)
    alpha1s = 0.5 + torch.rand(args.sample_mix)
    alpha2s = -0.1 - torch.rand(args.sample_mix)

    for comp_i, (dense, mean, alpha1, alpha2) in enumerate(zip(denseness, means,
            alpha1s, alpha2s)):
        A = densify_adj(adj_matrix,dense)
        gmrf_sample, _, _ = sample_gmrf(A, alpha1, alpha2, layers=1,
                diag_eps=args.diag_eps, mean=mean)
        gmrf_samples.append(gmrf_sample)
        print("Component {}: denseness={}, mean={:.3f}, alpha1={:.3f}, alpha2={:.3f}"\
                .format(comp_i, dense, mean, alpha1, alpha2))

    coeffs = 2.8*torch.randn(args.sample_mix)
    print("Weights: {}".format(coeffs))
    weighted_samples = torch.cat(gmrf_samples, axis=1)*coeffs # Linear combination
    final_sample = torch.sum(weighted_samples, axis=1, keepdim=True)
    graph_x.x = final_sample
elif args.prec_mix:
    # Construct a random precision matrix and draw one sample from it
    prec_mats = []
    alpha1s = 0.5 + torch.rand(args.prec_mix) # ~ Uniform([0.5,1.5])
    alpha2s = -0.1 - torch.rand(args.prec_mix) # ~ Uniform([-1.1,-0.1])

    for dense, (alpha1, alpha2) in enumerate(zip(alpha1s, alpha2s)):
        A = densify_adj(adj_matrix, dense)

        prec = construct_prec(A, alpha1, alpha2, 1)
        prec_mats.append(prec)
        print("Denseness={}, alpha1={:.3f}, alpha2={:.3f}".format(dense+1,
            alpha1, alpha2))

    precision_matrix = torch.sum(torch.stack(prec_mats, dim=0), dim=0)
    gmrf_sample, cov_matrix = sample_using_prec(precision_matrix,
            args.diag_eps, 0.) # Zero mean
    graph_x.x = gmrf_sample
    expl_gmrf = True

else:
    # Normal simulation, draw one sample from described GMRF
    mean = MEAN_STD*torch.randn(1)*(1 - args.zero_mean)
    gmrf_sample, precision_matrix, cov_matrix = sample_gmrf(adj_matrix,
            args.alpha1, args.alpha2, args.layers, args.diag_eps, mean)
    graph_x.x = gmrf_sample # Sample latent field x
    expl_gmrf = True

# Create Mask
if args.random_mask:
    n_mask = int(args.mask_fraction*graph_x.num_nodes)
    print("Masked out {} nodes".format(n_mask))
    unobs_indexes = torch.randperm(graph_x.num_nodes)[:n_mask]
    inv_node_mask = torch.zeros(graph_x.num_nodes).to(bool)
    inv_node_mask[unobs_indexes] = True
else:
    node_mask_components = torch.bitwise_and((graph_pos.pos >= MASK_LIMITS[0]),
            (graph_pos.pos < MASK_LIMITS[1]))
    inv_node_mask = torch.bitwise_and(node_mask_components[:,0],
            node_mask_components[:,1])

    graph_x.mask_limits = MASK_LIMITS

mask = torch.logical_not(inv_node_mask) # Mask is true for observed
graph_x.mask = mask

# Sample y|x (assumed Gaussian noise => posterior is GRMF)
y_noise = args.noise_std*torch.randn(args.num_nodes,1)
graph_y = graph_x.clone() # Deep copy of x (including mask)
graph_y.x = graph_x.x + y_noise

graph_y.obs_precision = 1./(args.noise_std**2) * graph_y.mask

if args.plot:
    vis.plot_graph(graph_x, name="x", title="x", show=True)
    vis.plot_graph(graph_y, name="y", title="y", show=True)

    # Plot covariance matrix
    if expl_gmrf:
        fig, ax = plt.subplots(nrows=1, ncols=2)

        prec_plot = ax[0].imshow(precision_matrix)
        ax[0].set(title="Precision")
        cov_plot = ax[1].imshow(cov_matrix)
        ax[1].set(title="Covariance")

        fig.colorbar(prec_plot, ax=ax[0])
        fig.colorbar(cov_plot, ax=ax[1])
        plt.show()

to_save = {
    "graph_x": graph_x,
    "graph_y": graph_y,
}

if expl_gmrf:
    # Compute true posterior
    print("Computing true posterior ...")
    mask_diag = torch.diag(graph_y.mask.to(float))
    post_true_precision = precision_matrix + (1./(args.noise_std**2))*mask_diag
    post_true_cov = torch.inverse(post_true_precision)

    post_true_mean = post_true_cov @ ((1./(args.noise_std**2))*(
        graph_y.x * graph_y.mask.to(float).unsqueeze(1)))

    post_true_stds = torch.sqrt(torch.diag(post_true_cov) +
            (args.noise_std**2)).unsqueeze(1)

    graph_post_true_mean = ptg.data.Data(edge_index=graph_pos.edge_index,
            pos=graph_pos.pos, x=post_true_mean)
    graph_post_true_std = ptg.data.Data(edge_index=graph_pos.edge_index,
            pos=graph_pos.pos, x=post_true_stds)

    if args.plot:
        vis.plot_graph(graph_post_true_mean, name="true_post_mean",
                title="(True) Posterior Mean", show=True)
        vis.plot_graph(graph_post_true_std, name="true_post_std_dev",
                title="(True) Posterior Marginal Std.-dev.", show=True)

    to_save["graph_post_true_mean"] = graph_post_true_mean
    to_save["graph_post_true_std"] = graph_post_true_std

# Log-determinant pre-processing steps
utils.log_det_preprocess(graph_y, args.dist_weight, args.compute_eigvals,
        args.compute_dad_traces, args.dad_k_max, args.dad_samples)

if args.plot and args.compute_eigvals:
    plt.hist(graph_y.eigvals.numpy(), bins=50, range=(-1,1))
    plt.title("Histogram of eigenvalues")
    plt.show()

    if args.dist_weight:
        w_eig_np = graph_y.weighted_eigvals.numpy()
        plt.hist(w_eig_np, bins=50, range=(w_eig_np.min(), w_eig_np.max()))
        plt.title("Histogram of (weighted) eigenvalues")
        plt.show()

# Save dataset
print("Saving graphs ...")
full_ds_name = "{}{}".format(ds_name, args.seed)
if args.layers > 1:
    full_ds_name += "_{}_layers".format(args.layers)
if args.densify_layers > 1:
    full_ds_name += "_{}_densified".format(args.densify_layers)
if args.random_mask:
    full_ds_name += "_random"
if args.dist_weight:
    full_ds_name += "_weighted"
if args.compute_cholesky:
    full_ds_name += "_cholesky"

utils.save_graph_ds(to_save, args, full_ds_name)

print("Graphs saved")

