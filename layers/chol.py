import torch
import torch_sparse
import torch_geometric as ptg


"""
Steps to compute ELBO

1.) compute prior mean
2.) compute prior eta = L @ L^T @ mean
3.) compute posterior eta+ = eta + H^T @ R^{-1} @ y
3.) sample from variational distribution: 
    3.1.) sample z
    3.2.) compute u = L_hat @ z + eta+
    3.3.) solve linear system Lv = u
    3.4.) solve linear system L^Tx = v
4.) evaluate log p(x) and log p(y | x)
"""

class GMRF(torch.nn.Module):
    def __init__(self, config, num_nodes):
        super().__init__()

        self.chol = CholFactor(config)
        self.mean = torch.zeros(num_nodes) # TODO also learn mean!

    def prior_log_likelihood(self, x, graph):
        L = self.chol.get_factor(graph)
        LTx = self.chol.apply(L.t(), x - self.mean)
        xTPx = LTx.view(1, self.dim) @ LTx.view(self.dim, 1)

        ll = -0.5 * xTPx + self.chol.log_det(L)

        return ll

    def prior_eta(self, graph):
        L = self.chol.get_factor(graph)
        eta = self.chol.apply(L, self.chol.apply_T(L, self.mean))

        return eta

    def posterior_eta(self, graph):
        prior_eta = self.prior_eta(graph)
        posterior_eta = prior_eta + graph.mask * graph.obs_precision * graph.y

        return posterior_eta

class VI(ptg.nn.MessagePassing):
    def __init__(self, config):
        super(VI, self).__init__(aggr="add", node_dim=0)

        self.chol = CholFactor(config)

    def sample(self, n_samples, graph, posterior_eta):
        z = torch.randn(n_samples, graph.num_nodes)

        L = self.chol.get_factor(graph)
        rhs = self.chol.apply(L, z) + posterior_eta

        # solve Lv = rhs for v
        v = self.triangular_solve(L, rhs)

        # solve L^Tx = v
        x = self.triangular_solve(L.t(), v, backward=True)

        return x

    def triangular_solve(self, mat, rhs, backward=False):
        # mat should be sparse triangular matrix
        # solve mat @ x = rhs for x

        num_nodes = rhs.size(0)
        diagonal = mat.get_diag()
        L = mat.remove_diag()
        x = rhs / diagonal
        eye = torch_sparse.SparseTensor.eye(num_nodes)

        node_iter = torch.arange(num_nodes)
        if backward:
            node_iter = reversed(node_iter)

        # nodes have to be treated sequentially because of dependencies
        for i in node_iter:
            node_idx = int(i)
            sub_mat = L[[node_idx], :]

            aggr = self.propagate(sub_mat, x=x, diagonal=diagonal[node_idx])
            x = x - (eye[node_idx] * aggr).to_dense()

        return x

    def message_and_aggregate(self, adj_t, x, diagonal):
        # faster than message passing based on atomic operations
        return torch_sparse.matmul(adj_t, x.view(-1, 1), reduce=self.aggr) / diagonal


class TestSolve(ptg.nn.MessagePassing):
    def __init__(self):
        super(TestSolve, self).__init__(aggr="add", node_dim=0)

    def triangular_solve(self, mat, rhs, backward=False):
        # mat should be sparse triangular matrix
        # solve mat @ x = rhs for x

        num_nodes = rhs.size(0)
        diagonal = mat.get_diag()
        L = mat.remove_diag()
        x = rhs / diagonal
        eye = torch_sparse.SparseTensor.eye(num_nodes)

        node_iter = torch.arange(num_nodes)
        if backward:
            node_iter = reversed(node_iter)

        # nodes have to be treated sequentially because of dependencies
        for i in node_iter:
            node_idx = int(i)
            sub_mat = L[[node_idx], :]

            aggr = self.propagate(sub_mat, x=x, diagonal=diagonal[node_idx])
            x = x - (eye[node_idx] * aggr).to_dense()

        return x

    def message_and_aggregate(self, adj_t, x, diagonal):
        # faster than message passing based on atomic operations
        return torch_sparse.matmul(adj_t, x.view(-1, 1), reduce=self.aggr) / diagonal



class CholFactor(ptg.nn.MessagePassing):
    def __int__(self, config):
        super(CholFactor, self).__init__(aggr="add", node_dim=0)

        self.node_embedding = torch.nn.Linear(config["n_node_features"], config["n_hidden"])
        self.layers = torch.nn.ModuleList([CholLayer(config) for _ in range(config["n_layers"])])

        # TODO: for now ignore edge features for edge prediction, later add info such as distances?
        self.L_ij = torch.nn.Linear(2 * config["n_hidden"], 1) # + len(config["edge_features"]), 1)


    def get_factor(self, graph):
        # node features for prior: degree, pos, other covariates, ...
        # node features for variational distr: measurement noise, observation mask
        # edge attributes for prior: distance, direction, ...
        # edge attributes for variational distr: prior L_ij

        # get node representations
        h = self.node_embedding(graph.node_features)
        for layer in self.layers:
            h = layer(h, graph.edge_index, graph.edge_attr)

        # get values of L
        row, col = graph.adj_L.indices()
        # if self.use_edge_features:
        #     inputs = torch.cat([h[row], h[col], graph.adj_L.values()], dim=1)
        # else:
        inputs = torch.cat([h[row], h[col]], dim=1)
        values = self.L_ij(inputs)

        L = graph.adj_L.set_value(values)

        return L

    def apply(self, adj_L, x):
        # compute z = Lx
        # for message passing to compute Lx, we need to work with L^T
        aggr = self.propagate(adj_L.t(), x=x)

        return aggr

    def apply_T(self, adj_L, x):
        # compute z = L^Tx
        aggr = self.propagate(adj_L, x=x)

        return aggr

    def message_and_aggregate(self, adj_t, x):
        # faster than message passing based on atomic operations
        # adj_t: transposed sparse adjacency matrix
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)


    def log_det(self, adj_L):
        # get values on diagonal of L
        L_ii = adj_L.get_diag()
        log_det = torch.log(L_ii).sum()

        return log_det


class CholLayer(ptg.nn.MessagePassing):
    def __init__(self, config):
        super(CholLayer, self).__init__(aggr="add")

        self.edge_update = torch.nn.Sequential(
            torch.nn.Linear(2 * config["n_hidden"] + config["n_edge_features"], config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["n_hidden"], config["n_hidden"])
        )

        self.node_update = torch.nn.Sequential(
            torch.nn.Linear(2 * config["n_hidden"], config["n_hidden"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["n_hidden"], config["n_hidden"]),
        )


    def forward(self, x, edge_index, edge_attr):
        # compute next node embeddings

        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr) # Shape (n_nodes*n_graphs,1)

        x = self.node_update(torch.cat([x, aggr], dim=1))

        return x

    def message(self, x_i, x_j, edge_attr):
        # message from node j to node i
        m_ij = self.edge_update(torch.cat([x_i, x_j, edge_attr], dim=1))
        return m_ij

