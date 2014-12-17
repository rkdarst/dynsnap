from itertools import product
import math
import numpy
import random

# convenience function
rand = lambda rng: rng.uniform(0, 1)

def group_by_t(it):
    """Transform events iterator to group format.

    This transforms an iterator over (t, e) pairs to (t, [e0, e1, e2,
    ...])."""
    last_t = None
    last_events = [ ]
    for t, e in it:
        if t != last_t:
            yield last_t, last_events
            last_t = t
            last_events = [ ]
        last_events.append(e)
    yield last_t, last_events
def get_rng(seed):
    """Helper function to get random number generator state with seed."""
    return random.Random(seed)



def toy101(**kwargs):
    for t in range(0, 100):
        events = [0, 1, 2, 3, 4, 5]
        for e in sorted(events):
            yield t, e

def toy102(**kwargs):
    for t in range(0, 100):
        generation = t // 10
        events = range(generation*6, (generation+1)*6)
        for e in sorted(events):
            yield t, e

def toy103(seed=None, **kwargs):
    rng = get_rng(seed)
    n = 50
    s = 25
    for t in range(0, 100):
        generation = t // 10
        events = range(generation*n, (generation+1)*n)
        events2 = rng.sample(events, s)
        for e in sorted(events2):
            yield t, e


def periodic(N=10000, p=.2, q=.2, c_scale=.01,
             t_jump=None, t_max=1000, tau=200.,
             seed=None, **kwargs):
    """

    p: Activated edges exist exist with this probability
    q: Fraction of activated edges
    c_scale: How many edges change per step?
    """
    rng = get_rng(seed)
    c_min = 0.00

    events = list(range(N))
    edge_p = { }
    for e in events:
        edge_p[e] = p   if rand(rng) < q   else 0.0

    for t in range(t_max):
        # build up all time scales.
        for e in events:
            if rand(rng) < edge_p[e]:
                yield t, e

        # Randomly perturb some edges
        c = c_min + c_scale * (.5 - .5 * math.cos(2*math.pi * t / tau))
        if t == t_jump:
            c = 1.0
        #print t, c
        for e in events:
            if rand(rng) < c:
                edge_p[e] = p   if rand(rng) < q   else 0.0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="benchmark model to simulate",)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--p", type=float, default=.5)
    parser.add_argument("--q", type=float, default=.2)
    parser.add_argument("--tau", type=int)
    parser.add_argument("--c_scale", type=int)
    parser.add_argument("--grouped", action='store_true', help="group output by time")
    args = parser.parse_args()



    it = globals()[args.model](**args.__dict__)

    if args['grouped']:
        it = group_by_t(it)
        for t, events in it:
            print t, " ".join(str(x) for x in events)
    else:
        for t, e in it:
            print t, e
