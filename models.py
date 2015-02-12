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


def drift(N=1000, p=.2, c=.01,
          t_max=1000, seed=None,
          t_crit=(),
          c_func=None,
          p_func=None,
          **kwargs):
    rng = get_rng(seed)
    t_crit = set(t_crit) # critical times: all events change
    next_eid = [ N-1 ]  # events 0--(N-1) are in the initial set
    def nextevent():
        next_eid[0] += 1
        return next_eid[0]
    if c_func is None:
        c_func = lambda : c
    if p_func is None:
        p_func = lambda : p


    events = set(xrange(N))
    event_c = dict((e, c_func()) for e in events)
    event_p = dict((e, p_func()) for e in events)

    for t in xrange(t_max):
        # critical events - all events change *before* t
        if t in t_crit:
            events = set(nextevent() for _ in xrange(N))
            event_c = dict((e, c_func()) for e in events)
            event_p = dict((e, p_func()) for e in events)


        # Yield events that occur now.
        for e in events:
            if rand(rng) < event_p[e]:
                yield t, e

        # Change events
        changes = [ ]
        for e in list(events):
            if rand(rng) < event_c[e]:
                changes.append(e)
        for e in changes:
            events.remove(e)
            i = nextevent()
            events.add(i)
            del event_c[e] ; event_c[i] = c_func()
            del event_p[e] ; event_p[i] = p_func()


def periodic(N=10000, p=.2, q=.2, c_scale=.01,
             t_crit=(), t_max=1000, tau=200.,
             seed=None, **kwargs):
    """

    p: Activated edges exist exist with this probability
    q: Fraction of activated edges
    c_scale: How many edges change per step?
    """
    rng = get_rng(seed)
    c_min = 0.00
    t_crit = set(t_crit)

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
        if t+1 in t_crit:
            c = 1.0
        #print t, c
        for e in events:
            if rand(rng) < c:
                edge_p[e] = p   if rand(rng) < q   else 0.0


#
# Tools for theoretically modeling distributions
#
def Pe(p, dt):
    return 1.-((1.-p)**dt)
def J1(dt, Pe):
    return Pe(dt) / float(2-Pe(dt))
import functools
import operator
from math import exp, log
def product(it):
    return functools.reduce(operator.mul, it, 1)
def P_all(it):
    return product(p for p in it)
def P_any(it):
    return 1. - product((1.-p) for p in it)
    #return 1. - exp(sum(log(1.-p) for p in it))
def Pe_c(c, p, dt):
    #x = 1 - product((1-p*(1-c)**dt_) for dt_ in range(1, dt+1))
    x = P_any( p*(1.-c)**(dt_-1)  for dt_ in range(1, dt+1)  )
    return x
def Pe_c_2(c, p, dt):
    pass
def J1_c(c, p, dt):

    left  = P_any( p * (1.-c)**(dt_-1)    for dt_ in range(1, dt+1)  )
    right = P_any( p * (1.-c)**(dt_)      for dt_ in range(1, dt+1)  )
    #half = 1 - product( (1-(1-(1-c)**(dt+1))*p)  for dt_ in range(2, dt+1)  )
    half_L = P_any( p * (1-(1.-c)**(dt_-1))  for dt_ in range(2, dt+1) )
    half_R = P_any( p * (1-(1.-c)**(dt_))    for dt_ in range(1, dt+1) )

    #half_L *= .6
    #half_R *= .6

    isect = left*right
    union = float( half_L + half_R + left + right - left*right )

    N = 100000
    print dt, int(isect*N), int(union*N), \
              int((half_L+left)*N), int((half_R+right)*N)

    return isect / union



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="benchmark model to simulate",)
    parser.add_argument("--N", type=int)
    parser.add_argument("--t_max", type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--q", type=float)
    parser.add_argument("--tau", type=int)
    parser.add_argument("--c_scale", type=int)
    parser.add_argument("--grouped", action='store_true', help="group output by time")
    args = parser.parse_args()

    args_dict = dict((k,v) for k,v in args.__dict__.iteritems() if v is not None)
    it = globals()[args.model](**args_dict)

    if args.__dict__['grouped']:
        it = group_by_t(it)
        for t, events in it:
            print t, " ".join(str(x) for x in events)
    else:
        for t, e in it:
            print t, e
