import matplotlib.pyplot as plt
import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace

from bem_helmholtz_3d.main import bem


def test_bem():
    k = 1.0
    N = 10
    xp = array_namespace(np.array(0))

    def uin[TArray: Array](x: TArray) -> TArray:
        xp = array_namespace(x)
        return xp.exp(1j * k * x[..., 0])

    # (N, 2)
    vertices = xp.exp(2j * xp.pi * xp.arange(N) / N * 2)
    vertices = xp.stack([vertices.real, vertices.imag], axis=-1)
    # (N, 2)
    simplex = xp.stack([xp.arange(N), (xp.arange(N) + 1) % N], axis=-1)
    simplex_vertices = vertices[simplex, :]
    calc = bem(simplex_vertices=simplex_vertices, k=xp.asarray(k), uin=uin)

    x, y = xp.meshgrid(xp.linspace(-4, 4, 100), xp.linspace(-4, 4, 100), indexing="ij")
    points = xp.stack([x, y], axis=-1)
    u = calc.uscat(points)

    # heatmap
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, xp.real(u), shading="auto")
    ax.set_title("Real part of the total potential")
    fig.colorbar(c, ax=ax)
    fig.savefig("tests/2d.png")
