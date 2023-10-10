import random
import numpy as np
import copy
from typing import List, Tuple
from scipy.optimize import minimize
import torch

import cvxopt
import cvxpy as cp

def PCGrad(grads: List[Tuple[torch.Tensor]], losses=None, ray=None, reduction: str = "sum") -> torch.Tensor:
    grads = [[grad] for grad in grads]
    pc_grad = copy.deepcopy(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = sum(
                [
                    torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                    for grad_i, grad_j in zip(g_i, g_j)
                ]
            )
            if g_i_g_j < 0:
                g_j_norm_square = (
                    torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                )
                for grad_i, grad_j in zip(g_i, g_j):
                    grad_i -= g_i_g_j * grad_j / g_j_norm_square

    merged_grad = [sum(g) for g in zip(*pc_grad)]
    if reduction == "mean":
        merged_grad = [g / len(grads) for g in merged_grad]

    return merged_grad[0]


def CAGrad(grads, losses=None, ray=None, alpha=0.4, rescale=1):
    n_tasks = len(grads)
    grads = grads.t()

    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    # GG = torch.zeros((n_tasks, n_tasks))
    # for i in range(n_tasks):
    #     for j in range(n_tasks):
    #         GG[i][j] = torch.dot(grads[i], grads[j]).cpu()
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(n_tasks) / n_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, n_tasks).dot(A).dot(b.reshape(n_tasks, 1))
            + c
            * np.sqrt(x.reshape(1, n_tasks).dot(A).dot(x.reshape(n_tasks, 1)) + 1e-8)
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha**2)
    else:
        return g / (1 + alpha)


def IMTL(grads_list, losses=None, ray=None):
    grads = {}
    norm_grads = {}

    for i, grad in enumerate(grads_list):
        norm_term = torch.norm(grad)

        grads[i] = grad
        norm_grads[i] = grad / norm_term

    G = torch.stack(tuple(v for v in grads.values()))
    D = G[0,] - G[1:,]

    U = torch.stack(tuple(v for v in norm_grads.values()))
    U = U[0,] - U[1:,]
    first_element = torch.matmul(
        G[0,],
        U.t(),
    )
    try:
        second_element = torch.inverse(torch.matmul(D, U.t()))
    except:
        # workaround for cases where matrix is singular
        second_element = torch.inverse(
            torch.eye(len(grads_list) - 1, device=norm_term.device) * 1e-8
            + torch.matmul(D, U.t())
        )

    alpha_ = torch.matmul(first_element, second_element)
    alpha = torch.cat(
        (torch.tensor(1 - alpha_.sum(), device=norm_term.device).unsqueeze(-1), alpha_)
    )
    return sum([alpha[i] * grads[i] for i in range(len(grads_list))])


def MGDA(grads_list, losses=None, ray=None):
    n_grads = len(grads_list)
    sol, _ = MinNormSolver.find_min_norm_element(grads_list)
    return torch.stack([n_grads * sol[i] * grads_list[i] for i in range(n_grads)]).sum(
        0
    )


def EW(grads_list, losses=None, ray=None):
    return grads_list.sum(0)

def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]

def EPO(grads_list, losses, ray):
    n_grads = len(grads_list)
    lp = ExactParetoLP(m=n_grads, n=grads_list[0].shape[0], r=ray.cpu().numpy())
    
    G = grads_list
    GG_T = G @ G.T
    GG_T = GG_T.detach().cpu().numpy()

    numpy_losses = losses.detach().cpu().numpy()

    try:
        alpha = lp.get_alpha(numpy_losses, G=GG_T, C=True)
    except Exception as excep:
        print(excep)
        alpha = None

    if alpha is None:  # A patch for the issue in cvxpy
        alpha = (ray / ray.sum()).cpu().numpy()

    alpha *= n_grads
    alpha = torch.from_numpy(alpha).to(G.device)

    return sum([alpha[i] * grads_list[i] for i in range(len(grads_list))])
 

class ExactParetoLP(object):
    """modifications of the code in https://github.com/dbmptr/EPOSearch"""

    def __init__(self, m, n, r, eps=1e-4):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.n = n
        self.r = r
        self.eps = eps
        self.last_move = None
        self.a = cp.Parameter(m)  # Adjustments
        self.C = cp.Parameter((m, m))  # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)  # d_bal^TG
        self.rhs = cp.Parameter(m)  # RHS of constraints for balancing

        self.alpha = cp.Variable(m)  # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)  # objective for balance
        constraints_bal = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Simplex
            self.C @ self.alpha >= self.rhs,
        ]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Restrict
            self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
            self.C @ self.alpha >= 0,
        ]
        constraints_rel = [
            self.alpha >= 0,
            cp.sum(self.alpha) == 1,  # Relaxed
            self.C @ self.alpha >= 0,
        ]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0  # Stores the latest Optimum value of the LP problem
        self.mu_rl = 0  # Stores the latest non-uniformity

    def get_alpha(self, l, G, r=None, C=False, relax=False):
        r = self.r if r is None else r
        assert len(l) == len(G) == len(r) == self.m, "length != m"
        rl, self.mu_rl, self.a.value = adjustments(l, r)
        self.C.value = G if C else G @ G.T
        self.Ca.value = self.C.value @ self.a.value

        if self.mu_rl > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf  # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"

        return self.alpha.value


def mu(rl, normed=False):
    if len(np.where(rl < 0)[0]):
        raise ValueError(f"rl<0 \n rl={rl}")
        # return None
    m = len(rl)
    l_hat = rl if normed else rl / rl.sum()
    eps = np.finfo(rl.dtype).eps
    l_hat = l_hat[l_hat > eps]
    return np.sum(l_hat * np.log(l_hat * m))


def adjustments(l, r=1):
    m = len(l)
    rl = r * l
    l_hat = rl / rl.sum()
    mu_rl = mu(l_hat, normed=True)
    a = r * (np.log(l_hat * m) - mu_rl)
    return rl, mu_rl, a




class MinNormSolver:
    MAX_ITER = 20
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 5e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = (vecs[i] * vecs[j]).sum().item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = (vecs[i] * vecs[i]).sum().item()
                if (j, j) not in dps:
                    dps[(j, j)] = (vecs[j] * vecs[j]).sum().item()
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)]
                )
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        # print("torch.sort(y)", torch.sort(y)[0])
        sorted_y = torch.flip(torch.sort(y)[0], dims=[0])
        tmpsum = 0.0
        tmax_f = (y.sum() - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.max(y - tmax_f, torch.zeros(y.shape).cuda())

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (torch.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])
        t = 1

        if len(tm1[tm1 > 1e-7]) > 0:
            t = (tm1[tm1 > 1e-7]).min()
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, (tm2[tm2 > 1e-7]).min())

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """

        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = torch.zeros(n).cuda()
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = torch.zeros((n, n)).cuda()
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * torch.mm(grad_mat, sol_vec.view(-1, 1)).view(-1)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)

            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            # print("Change: ", change)
            try:
                if change.pow(2).sum() < MinNormSolver.STOP_CRIT:
                    return sol_vec, nd
            except Exception as e:
                print(e)
                print("Change: ", change)
                # return sol_vec, nd
                pass
            sol_vec = new_sol_vec
        return sol_vec, nd


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == "l2":
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == "loss":
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == "loss+":
        for t in grads:
            gn[t] = losses[t] * np.sqrt(
                np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]])
            )
    elif normalization_type == "none":
        for t in grads:
            gn[t] = 1.0
    else:
        print("ERROR: Invalid Normalization Type")
    return gn