import bempp_cl.api
import numpy as np
from bempp_cl.api.grid import union
from bempp_cl.api.operators.boundary.helmholtz import single_layer
from bempp_cl.api.shapes import sphere
from jordan_form import multiplicity
from jordan_form.plot import plot_eigval_with_multiplicity
from matplotlib import pyplot as plt
from ss_hankel import ss_h_circle
from tqdm import tqdm

# http://www.bempp.org/operators_and_potentials.html
centers =[[ 0.        ,  0.        ,  0.        ],
       [ 0.75646074, -3.86284462,  0.        ],
       [ 4.42185842, -1.09750114,  0.        ]]
radii = [1.0411, 0.9448001, 0.9448001]
grid = union([sphere(h=0.3, origin=center, r=radius) for center, radius in zip(centers, radii)])
piecewise_const_space = bempp_cl.api.function_space(grid, "DP", 0)
k = 1


def f(ks, /):
    result = []
    for k in tqdm(ks):
        slp = single_layer(piecewise_const_space, piecewise_const_space, piecewise_const_space, k)
        slp_discrete = slp.strong_form()
        mat = bempp_cl.api.as_matrix(slp_discrete)
        result.append(mat)
    return np.stack(result, axis=0)


res = ss_h_circle(
    f,
    num_vectors=12,
    max_order=12,
    circle_n_points=4096,
    circle_center=3.5-1j,
    circle_radius=0.9,
)
print(res)
fig, ax = plt.subplots()
plot_eigval_with_multiplicity(
    multiplicity(
        res.eigval,
        res.eigvec,
        atol_algebraic=1e-3,
        atol_geometric=1e-3,
        rtol_algebraic=1e-3,
        rtol_geometric=1e-3,
    ),
)
fig.savefig("tests/eigval_multiplicity.png")
