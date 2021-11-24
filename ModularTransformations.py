# -*- coding: utf-8 -*-
from math import sqrt, cos, sin, tan
from sympy.combinatorics.free_groups import free_group
from itertools import product
import numpy as np
from functools import reduce
from planegeometry.geometry import Mobius, Circle, Line # https://github.com/stla/PyPlaneGeometry
import pyvista as pv


F, a, b = free_group("a, b")
B = b.inverse()
elems = {"a": a, "b": b, "B": B}
keys = list(elems.keys())

x = list(product(keys, repeat=6)) + list(product(keys, repeat=7))

def multiply(list_of_symbols):
    elements = list(map(lambda k: elems[k], list_of_symbols))
    return reduce(lambda u, v: u*v, elements)

def unique_with(L, f):
    size = len(L)
    for i in range(size-1):
        j = i + 1
        while j < size:
            if f(L[i], L[j]):
                del L[j]
                size -= 1
            else:
                j += 1
    return L[:size]

Gn = unique_with(list(map(multiply, x)), lambda u, v: u == v)

def total(g):
    dec = g.array_form
    powers = list(map(lambda x: abs(x[1]), g))
    return sum(powers)

totals = list(map(total, Gn))

sizes = np.asarray(totals, dtype=int)
indices = np.where(np.less_equal(sizes, 5))[0]
Gn = [Gn[i] for i in indices.tolist()]

T = Mobius(np.array([[0, -1], [1, 0]]))
U = Mobius(np.array([[1, 1], [0, 1]]))
Uinv = Mobius(np.array([[1, -1], [0, 1]]))

Mobs = {
    "a": T,
    "b": U,
    "B": Uinv
}

def transfo2word(g):
    tup = g.array_form
    word = ""
    for j, (t, i) in enumerate(tup):
        t = str(t)
        if i < 0:
            i = -i
            t = str.upper(t) if t == "b" else "a"
        word = word + t*i
    return word.replace("aa", "") 
        
allwords = np.unique([transfo2word(g) for g in Gn], axis=0)

        
def word2seq(word):
    return [*word]

def seq2mobius(seq):
    if len(seq) == 0:
        return Mobius([[1, 0], [0, 1]])
    if len(seq) == 1:
        return Mobs[seq[0]]
    mobs = [Mobs[s] for s in seq]
    return reduce(lambda m1, m2: m1.compose(m2), mobs)
        
c0 = Circle((0, 1.5), 0.5)
Phi = Mobius([[1j, 1], [1, 1j]])

allcircles = [None]*len(allwords)
for i, word in enumerate(allwords):
    M = seq2mobius(word2seq(word)).compose(Phi)
    allcircles[i] = M.transform_circle(c0)
allcircles = unique_with(allcircles, lambda c1, c2: c1.is_equal(c2))

Psi = Phi.inverse()
R = U.compose(T)
def Rt(t):
    return R.gpower(t)

pltr = pv.Plotter()
for i, circ in enumerate(allcircles):
    M = Psi.compose(Rt(1)).compose(Phi)
    circ = M.transform_circle(circ)
    cx, cy = circ.center
    sphere = pv.Sphere(circ.radius, center=(cx, cy, 0))
    pltr.add_mesh(sphere, color="navy", smooth_shading=True, specular=20)
    pltr.camera_position = "xy"
pltr.show()