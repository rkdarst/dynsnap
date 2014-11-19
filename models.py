from itertools import product
import math
import numpy
import random

import networkx

rand = lambda: random.uniform(0, 1)

N = 100
nodes = set(range(N))

pmatrix = numpy.zeros((len(nodes), len(nodes)), dtype=float)

p = .5   # Activated edges exist exist with this probability
q = .2   # Fraction of activated edges
c_scale = .02  # How many edges change per step?
c_min = 0.00
tau = 100.
t_jump = 300
t_max = 1000

edges = [ (n1, n2) for n1 in nodes for n2 in nodes if n2 < n1 ]

edge_p = { }
for n1, n2 in edges:
    edge_p[(n1, n2)] = p   if rand() < q   else 0.0

import tnetsql

tnet = tnetsql.TNet(':memory:')
for t in range(t_max):
    # build up all time scales.
    tnet.add_edges((n1, n2, t, 1.0) for n1,n2 in edges
                   if rand() < edge_p[(n1, n2)])

    # Randomly perturb some edges
    c = c_min + c_scale * (.5 - .5 * math.cos(2*math.pi * t / tau))
    if t == t_jump:
        c = 1.0
    #print t, c
    for n1, n2 in edges:
        if rand() < c:
            edge_p[(n1, n2)] = p   if rand() < q   else 0.0

print len(tnet)
