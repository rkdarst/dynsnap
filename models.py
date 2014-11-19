from itertools import product
import math
import numpy
import random

rand = lambda: random.uniform(0, 1)

def toy101(args):
    for t in range(0, 100):
        events = [0, 1, 2, 3, 4, 5]
        for e in events:
            print t*100, e, 1

def toy102(args):
    for t in range(0, 100):
        generation = t // 10
        events = range(generation*6, (generation+1)*6)
        for e in events:
            print t*100, e, 1

def toy103(args):
    n = 50
    s = 25
    for t in range(0, 100):
        generation = t // 10
        events = range(generation*n, (generation+1)*n)
        events2 = random.sample(events, s)
        for e in events2:
            print t*100, e, 1


def periodic(args):

    p = .5   # Activated edges exist exist with this probability
    q = .2   # Fraction of activated edges
    c_scale = .02  # How many edges change per step?
    c_min = 0.00
    tau = 100.
    t_jump = 300
    t_max = 1000

    events = list(range(args.NN))
    edge_p = { }
    for e in events:
        edge_p[e] = p   if rand() < q   else 0.0

    for t in range(t_max):
        # build up all time scales.
        for e in events:
            if rand() < edge_p[e]:
                print t, e

        # Randomly perturb some edges
        c = c_min + c_scale * (.5 - .5 * math.cos(2*math.pi * t / tau))
        if t == t_jump:
            c = 1.0
        #print t, c
        for e in events:
            if rand() < c:
                edge_p[e] = p   if rand() < q   else 0.0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="benchmark model to simulate",)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--p", type=float, default=.5)
    parser.add_argument("--q", type=float, default=.2)
    parser.add_argument("--tau", type=int)
    parser.add_argument("--c_scale", type=int)
    args = parser.parse_args()


    globals()[args.model](args)
