# Richard Darst, November 2014

from __future__ import print_function, division

import argparse
import collections
import datetime
import heapq
from itertools import product
import math
from math import floor, log, log10, sqrt
import numpy
import os
import random
import sqlite3
import sys
import time

# Python 2/3 compatibility.  Avoid dependency on six for one function.
if sys.version_info.major < 3:
    def iteritems(x): return x.iteritems()
else:
    def iteritems(x): return iter(x.items())

from events import Events, load_events

# This is used to hold output, but it is not really used anymore.
ResultsRow = collections.namedtuple('ResultsRow',
                                    ('tlow', 'thigh', 'dt', 'x_max', 'measure_data',
                                     'finder_data'))

# The following implement weighed sets.
class _LenProxy(object):
    def __init__(self, l):   self.l = l
    def __len__(self): return self.l
class WeightedSet(object):
    """Weighted set implementation, emulating set()"""
    def __init__(self, it):
        data = self._data = { }
        for x, w in it:
            if x not in data:    data[x] = w
            else:                data[x] += w
    @classmethod
    def _from_data(cls, data):
        """Directly create a WeightedSet from a data dict"""
        self = cls([])
        self._data = data
        return self
    def update(self, it):
        """Like set.update() but add weights"""
        data = self._data
        for x, w in it:
            if x not in data:    data[x] = w
            else:                data[x] += w
    def __len__(self):
        """Number of elements in set"""
        return len(self._data)
    def __and__(self, other):
        """Set intersection"""
        if len(self) <= len(other):
            A, B = self._data, other._data
        else:
            A, B = other._data, self._data
        # A is the smaller set
        intersection = 0.0
        for x, w in iteritems(A):
            if x in B:
                intersection += min(A[x], B[x])
        return _LenProxy(intersection)
    def __or__(self, other):
        """Set union"""
        if len(self) <= len(other):
            A, B = self._data, other._data
        else:
            A, B = other._data, self._data
        # A is the smaller set
        union = sum(B.itervalues())
        for x, w in iteritems(A):
            union += max(0,   A[x] - B.get(x, 0))
        return _LenProxy(union)
    def union(self, other):
        """Set union, return new set"""
        if len(self) <= len(other):
            A, B = self._data, other._data
        else:
            A, B = other._data, self._data
        data = dict(B)
        # A is the smaller set
        for x, w in iteritems(A):
            if x not in data:    data[x] = w
            else:                data[x] += w
        return self._from_data(data)
    def dot(self, other):
        """Dot product of two sets (as vectors)"""
        if len(self._data) <= len(other._data):
            A, B = self._data, other._data
        else:
            A, B = other._data, self._data
        # A is the smaller set
        return sum(w*B.get(x, 0.0) for x, w in iteritems(A))
    def dot_uw(self, other):
        """Dot product of two sets (as vectors), unweighted"""
        if len(self._data) <= len(other._data):
            A, B = self._data, other._data
        else:
            A, B = other._data, self._data
        # A is the smaller set
        return sum(1 for x, w in iteritems(A) if x in B)
    def norm(self):
        """Norm of this set (as vector)"""
        return sqrt(sum(w*w for w in self._data.itervalues()))


def test_weightedset():
    """Small test function to test weighted sets"""
    from nose.tools import assert_equal
    A = WeightedSet([('a',1), ('b',2),])
    B = WeightedSet([('b',1), ('c',2),])
    C = WeightedSet([('a',1), ('c',1), ('d',3)])
    assert_equal(len(A & B), 1)
    assert_equal(len(A | B), 5)
    assert_equal(len(A & C), 1)
    assert_equal(len(A | C), 7)
def approxeq(x, y, d=1e-6):
    """Approximately equal to function, to a tolerance d"""
    if x+y==0: return x==y
    return 2*abs(x-y)/(x+y) < d



class SnapshotFinder(object):
    """This is the core snapshot finder class.  Basic usage:

    evs = events.Events()  # or load events
    finder = SnapshotFinder(evs, ...)
    while True:
        out = finder.find()
        if out is None: break

    In most cases, you can use the main() function which handles much
    more of the set up and data collection.
    """
    quiet = False
    old_es = None   # None on first round
    old_incremental_es = None
    dt_min = None
    dt_max = None
    dt_step = 1
    log_dt_min = None
    log_dt_max = None
    dt_first_override = None
    merge_first = False
    peakfinder = 'longest'
    measure = 'jacc'
    dtmode = 'log'
    find_critical_events = True

    dt_pastpeak_factor = 25
    dt_peak_factor = 0.0
    dt_pastpeak_min = 0
    dt_pastpeak_max = None
    dt_search_min = 0


    def __init__(self, evs, tstart=None, tstop=None, weighted=False,
                 measure=None,
                 dtmode=None, peakfinder=None,
                 merge_first=None,
                 args={},
                 dt_min=None, dt_max=None, dt_step=None,
                 log_dt_min=None, log_dt_max=None,
                 dt_pastpeak_factor=None, dt_peak_factor=None,
                 dt_pastpeak_min=None, dt_pastpeak_max=None,
                 dt_search_min=0,
                 find_critical_events=None,
                 quiet=None,
                 ):
        self.evs = evs
        if isinstance(args, argparse.Namespace):
            args = args.__dict__
        self.args = args
        self.weighted = weighted
        self.last_dt_max = 0

        if len(evs) == 0:
            raise ValueError("There are no events within our Events object!")

        try:
            if measure is None: measure = self.measure
            self.measure = getattr(self, 'measure_'+measure)
        except AttributeError:
            raise ValueError("Unknown measure: %s"%measure)

        if tstart is not None:    self.tstart = tstart
        else:                     self.tstart = evs.t_min()
        if tstop is not None:     self.tstop  = tstop
        else:                     self.tstop  = evs.t_max()

        try:
            if dtmode is None: dtmode = self.dtmode
            self.iter_all_dts = getattr(self, 'iter_all_dts_'+dtmode)
        except AttributeError:
            raise ValueError("Unknown dtmode: %s"%dtmode)

        try:
            if peakfinder is None: peakfinder = self.peakfinder
            self.pick_best_dt = getattr(self, 'pick_best_dt_'+peakfinder)
        except AttributeError:
            raise ValueError("Unknown peakfinder: %s"%peakfinder)

        locals_ = locals()
        for name in ('dt_min', 'dt_max', 'dt_step',
                     'log_dt_min', 'log_dt_max',
                     'merge_first',
                     'dt_pastpeak_factor',
                     'dt_peak_factor',
                     'dt_pastpeak_min',
                     'dt_pastpeak_max',
                     'dt_search_min',
                     'find_critical_events',
                     'quiet',
                     ):
            if locals_[name] is not None:
                setattr(self, name, locals_[name])
        if self.dt_min   is None:    self.dt_min   = self.dt_step
        if self.dt_max   is None:    self.dt_max   = 1000*self.dt_step


    def run(self, verbose=False, **kwargs):
        """Do a run, return results"""
        results = Results(self, args=kwargs)
        while True:
            x = self.find()
            if x is None:
                break
            results.add(self)
            # Printing if desired
            if verbose:
                tlow  = x[0]
                thigh = x[1]
                dt = thigh-tlow
                val = self.found_x_max
                print(tlow, thigh, dt, val, len(self.old_es))
        return results



    # Two generalized set-making functions.  These take an iterator
    # over events, and return set objects.  They are separate methods,
    # so that we can have either unweighted or weighted sets, or even
    # higher-level ideas.  Returned objects should be able to do
    # len(a|b), len(a&b), and a.union(b)
    def _set_make(self, cursor):
        if not self.weighted:
            es = set(e for t, e, w in cursor)
        else:
            es = WeightedSet((e,w) for t, e, w in cursor)
        return es
    def _set_update(self, set_, cursor):
        if not self.weighted:
            set_.update(set(e for t,e,w in cursor))
        else:
            set_.update(set((e,w) for t,e,w in cursor))

    # Interval-getting functions.  These use the internal state
    # `self.tstart` and the argument dt to return edge sets (made
    # using _make_set).
    def get_first(self, dt):
        """Get a first interval: return the two intervals.

        This uses self.tstart to make to intervals using dt."""
        #cursor = self.evs[self.tstart: self.tstart+dt]
        #es1b = self._set_make(cursor)
        # cached version
        es1 = self.get_succesive(dt)
        #assert es1 == es1b

        cursor = self.evs[self.tstart+dt: self.tstart+2*dt]
        es2 = self._set_make(cursor)
        return es1, es2
    def get_succesive(self, dt):
        """Get a successive interval.

        self.old_es should be provided."""
        es = None
        if self.old_incremental_es is not None:
            # We can try to use the old edge set to save recreating
            # everything.  We only add the new new edges between
            # old_dt and dt.
            old_tstart, old_dt, old_es = self.old_incremental_es
            if old_tstart == self.tstart and old_dt < dt:
                #print "using cache"
                cursor = self.evs[self.tstart+old_dt : self.tstart+dt]
                self._set_update(old_es, cursor)
                es = old_es
        if es is None:
            # Recreate edge set from scratch
            es = self.evs[self.tstart: self.tstart+dt]
            es = self._set_make(es)

        # Cache our results for the next interval
        self.old_incremental_es = (self.tstart, dt, es)

        return es
    def get(self, dt):
        """Calls either get_first or get_succesive.

        Also, for successive intervals, returns (old_es, next_es)."""
        if self.old_es == None:
            return self.get_first(dt)
        else:
            return self.old_es, self.get_succesive(dt)

    # Different measurement functions: either Jaccard or NMI.  (Note:
    # NMI is old, was for graphs which is no longer supported).
    def measure_jacc(self, es1s, es2s):
        """Jaccard similarity of event sets.  Weighted or unweighted."""
        union = (es1s | es2s).__len__()
        if union == 0:
            x = float('nan')
            self._measure_data = (0, 0, es1s.__len__(), es2s.__len__())
        else:
            intersect = (es1s & es2s).__len__()
            alpha = 1
            x = intersect / float(alpha*(union-intersect) + intersect)
            self._measure_data = (intersect, union, es1s.__len__(), es2s.__len__())
        return x
    def measure_nmi(self, es1s, es2s):
        """NMI similarity of event sets.  Graphs only, *not* implemented now."""
        g1 = nx.Graph(x for x in es1s)
        g2 = nx.Graph(x for x in es2s)
        g1lcc = g1.subgraph(nx.connected_components(g1)[0])
        g2lcc = g2.subgraph(nx.connected_components(g2)[0])
        cdargs = dict(verbosity=0)
        c1 = algs.Louvain(g1lcc, **cdargs).cmtys
        c2 = algs.Louvain(g2lcc, **cdargs).cmtys
        #c1, c2 = pcd.cmtycmp.limit_to_overlap(c1, c2)
        #nmi = pcd.cmtycmp.nmi(c1, c2)
        nmi = pcd.cmtycmp.F1_python2(c1, c2)
        return nmi
    def measure_cosine(self, es1s, es2s):
        """Cosine similarity of event sets.  Weighted sets only."""
        dot = es1s.dot(es2s)
        norm1 = es1s.norm()
        norm2 = es2s.norm()
        cossim = dot/(norm1*norm2)
        self._measure_data = (dot, norm1, norm2, len(es1s), len(es2s))
        return cossim
    def measure_cosine_uw(self, es1s, es2s):
        """Cosine similarity of event sets, as unweighted sets."""
        dot = len(es1s & es2s)
        cossim = dot/sqrt(len(es1s) * len(es2s))
        self._measure_data = (dot, len(es1s), len(es2s))
        return cossim


    def iter_all_dts_linear(self):
        """Get all dt values we will search."""
        # Create our range of dts to test.
        #all_dts = numpy.arange(self.dt_min, self.dt_max+self.dt_step,
        #                       self.dt_step)
        # sqlite3 can't handle numpy.int64, convert all to floats.
        #all_dts = [ eval(repr(x)) for x in all_dts ]
        #return all_dts
        dt = self.dt_min
        stop = self.dt_max+self.dt_step
        step = self.dt_step
        while True:
            yield dt
            if dt > stop: break
            dt += step
            #if self.tstart + dt > self.tstop: break  # moved to find()
    def iter_all_dts_log(self):
        # Find the next event to set the log scale properly.
        c = self.evs.conn.cursor()
        c.execute('SELECT DISTINCT t FROM event WHERE t > ? '
                  'ORDER BY t ASC LIMIT 1', (self.tstart, ))
        smallest_dt = c.fetchone()[0] - self.tstart
        c.close()
        # set this to minimum dt we scan.  Should be a power of ten
        # (10**(int)).
        if self.log_dt_min is not None:
            base_scale = self.log_dt_min
        # Find a base scale smaller than our next event.
        else:
            base_scale = 10.**floor(log10(smallest_dt))
        # Specifies how many digits we scan in the log thing.
        # 0 = 1,2,3,..10,20,..100,200
        #-1 = 1,2,..10,11...100,110,333
        log_precision = 1

        i = 1
        while True:
            dt = i * base_scale
            yield dt
            i += int(10**( max(0, int(log10(i))-log_precision)  ))
            #if self.log_dt_max and dt > self.log_dt_max: break
            #if self.tstart + dt > self.tstop: break  # moved to find()
    def iter_all_dts_event(self):
        """Iterate dts that actually exist.

        This is the most adaptive sampling method, but is bad when the
        intrinsic time is much greater than the shortest event time.
        In that case, set dtstep and use linear mode."""
        tstart = self.tstart
        stop = self.dt_max
        c = self.evs.conn.cursor()
        c.execute('SELECT DISTINCT t FROM event WHERE t >= ? '
                  'ORDER BY t ASC', (self.tstart, ))
        for row in c:
            yield row[0] - tstart
            if row[0] - tstart > stop: break
    iter_all_dts = iter_all_dts_linear

    # Below, we have the --peakfinder related options.  These relate
    # to picking the maximum value out of the plateau, and
    # implementing stop conditions.
    class StopSearch(BaseException):
        """Relay a stop condition from the code to several layers up.
        """
        def __init__(self, i_max, *args, **kwargs):
            self.i_max = i_max
            return super(self.__class__, self).__init__(*args, **kwargs)

    def pick_best_dt_shortest(self, dt, dts, xs):
        """--peakfinder=shortest"""
        i_max = numpy.argmax(xs)
        dt_peak = dts[i_max]

        dt_extra_ = max(self.dt_pastpeak_factor*dt_peak,
                        self.dt_pastpeak_factor*self.last_dt_max,
                        self.dt_pastpeak_min)

        # If we have found a peak, and the value has now decreased to
        # J_peak*dt_peak_factor, stop the search.
        if (xs[-1] < xs[i_max]*self.dt_peak_factor
                      and len(dts) > 10
                      and dt>=self.dt_search_min):
            raise self.StopSearch(i_max)
        # If we have searched to (dt_peak + dt_extra_), then stop
        # search.  dt_extra_ is a dynamic quantity that encapsulates
        # "a certain factor past the current peak."
        if (len(dts) > 10
                    and dt > dt_peak + dt_extra_
                    and dt>=self.dt_search_min):
            raise self.StopSearch(i_max)
        # If we have gone beyond self.dt_pastpeak_max past the max,
        # then unconditionally stop the search.
        if self.dt_pastpeak_max and dt > dt_peak + self.dt_pastpeak_max:
            raise self.StopSearch(i_max)
        return i_max
    def pick_best_dt_longest(self, dt, dts, xs):
        """--peakfinder=longest"""
        xs_array_reversedview = numpy.asarray(xs)[::-1]
        i_max_reversed = numpy.argmax(xs_array_reversedview)
        i_max = len(xs) - 1 - i_max_reversed
        dt_peak = dts[i_max]

        dt_extra_ = max(self.dt_pastpeak_factor*dt_peak,
                        self.dt_pastpeak_factor*self.last_dt_max,
                        self.dt_pastpeak_min)

        # If we have found a peak, and the value has now decreased to
        # J_peak*dt_peak_factor, stop the search.
        if (xs[-1] < xs[i_max]*self.dt_peak_factor
                      and len(dts) > 10
                      and dt>=self.dt_search_min):
            raise self.StopSearch(i_max)
        # If we have searched to (dt_peak + dt_extra_), then stop
        # search.  dt_extra_ is a dynamic quantity that encapsulates
        # "a certain factor past the current peak."
        if (len(dts) > 10
                    and dt > dt_peak + dt_extra_
                    and dt>=self.dt_search_min):
            raise self.StopSearch(i_max)
        # If we have gone beyond self.dt_pastpeak_max past the max,
        # then unconditionally stop the search.
        if self.dt_pastpeak_max and dt > dt_peak + self.dt_pastpeak_max:
            raise self.StopSearch(i_max)
        return i_max
    def pick_best_dt_greedy(self, dt, dts, xs):
        """--peakfinder=greedy"""
        # if we have decreased on the last step AND if we have
        # exceeded dt_search_min.
        if len(xs) > 2 and xs[-2] > xs[-1] and dt>self.dt_search_min:
            raise self.StopSearch(len(xs)-2)
        # break condition when exceeding total available time.
        if self.tstart + dt > self.tstop:
            raise self.StopSearch(len(xs)-1)

        return len(xs) - 1
    pick_best_dt = pick_best_dt_longest

    # This is the core function that does a search.
    def find(self):
        """Core snapshot finding method.

        Returns (low, high) values.  This method can be repeatedly
        called to do a segmentation, but most real use actually calls
        the main() function.

        Internal documentation about how this method works:

        iter_all_dts(): Gives dt values to check.  Repeats indefinitely,
        break happens in pick_best_dt().
        - dt_min: starting dt (linear)
        - dt_max: ending dt (linear)
        - dt_step: step size (linear)
        - log_dt_min: minimum log step size (log)
        - (event: no parameters)

        pick_best_dt(): picks the optimal dt value.
        - Options are pick_best_dt_shortest, pick_best_dt_longest,
          pick_best_dt_greedy.  The following options are defined:
        """
        # Our stop condition.  Returning None is a sentinel to the
        # caller stop the analysis.
        if self.tstart >= self.tstop:
            return None
        # Do not do a search for the first interval but just select something.
        if self.old_es is None and self.dt_first_override:
            self.t_crit = False
            self._finder_data = dict(xs=[], ts=[], dts=[],
                                        measure_data=[])
            es1s, es2s = self.get(self.dt_first_override)
            x = self.measure(es1s, es2s)
            self.found_x_max = x
            old_tstart = self.tstart
            self.interval_low = old_tstart
            self.tstart = self.tstart + self.dt_first_override
            self.interval_high = self.tstart
            self.old_n_events = self.evs.count_interval(old_tstart, self.tstart)
            if self.old_es is None:
                self.old_es = es1s
            return self.interval_low, self.interval_high



        dts = [ ]
        xs = [ ]
        self._finder_data = dict(xs=[], ts=[], dts=[],
                                 measure_data=[])
        i_max = None
        try:
          break_next_round = False
          for i, dt in enumerate(self.iter_all_dts()):
            # Condition for breaking.  This assumes that the dt values
            # are monitonically increasing.  If not, the iter_all_dts
            # method needs to ensure that this condition is never
            # fulfilled until after it is ready to stop.
            # We need to break _after_ the loop, because we must test
            # one value greater than the stopping point at least
            # because the upper interval is open.
            if break_next_round:
                break
            if self.tstart + dt > self.tstop:
                break_next_round = True
                continue
            # Get our new event sets for old and new intervals.
            es1s, es2s = self.get(dt)
            # Skip if there are no events in either interval (we don't
            # expect most measures to be defined in this case.)
            if len(es1s) == 0 or len(es2s) == 0:
                continue
            x = self.measure(es1s, es2s)
            dts.append(dt)
            xs.append(x)

            # Store data for later plotting.
            self._finder_data['dts'].append(dt)
            self._finder_data['ts'].append(self.tstart+dt)
            self._finder_data['xs'].append(x)
            self._finder_data['measure_data'].append(self._measure_data)

            # Find best dt.  This can raise self.StopSearch in order
            # to terminate the search early (in that case it should
            # set i_max as an attribute of the exception.
            i_max = self.pick_best_dt(dt, dts, xs)
        except self.StopSearch as e:
            i_max = e.i_max

        # We are completly done with the interval.  break the loop.
        if i_max is None:
            return None


        #assert xs[i_max] != xs[-1], "Plateaued value"
        #assert i_max != len(xs)-1, "Continually increasing value"

        # In the case below, we have a continually increasing jacc,
        # which indicates that there was some extreme event.  In this
        # case, we restart completly.
        self.t_crit = False
        if ((.95*xs[i_max] <= xs[-1])
            and self.old_es is not None
            and (self.tstart+dts[-1] < self.tstop-.001
                 or approxeq(xs[len(xs)//2], xs[-1], .01))
            and self.find_critical_events):
            if not self.quiet:
                print("### critical event detected at t=%s"%self.tstart)
            # At this point, we have had an extreme event and we
            # should restart from zero.
            # Save some old values to use when recalculating things.
            old_tstart = self.tstart
            prev_es = self.old_es
            # Remove old interval.  We need a fresh restart as in the
            # beginning of the calculation.
            self.old_es = None
            # Do the actual finding.  Save return since that is the
            # signature we need to return in the end.
            ret_val =  self.find()
            self.t_crit = True
            # The following things need to be re-calculated since the
            # eself.find is comparing the two initial intervals, and
            # not the previos interval and this interval.  Note that
            # not everything is being updated!  So far, nly
            # self.found_x_max is.
            self.found_x_max = self.measure(prev_es, self.old_es)
            #self.last_dt_max = self.tstart - old_tstart # not updated
            return ret_val


        #print xs
        dt_max = self.found_dt_max = dts[i_max]
        x_max  = self.found_x_max  = xs[i_max]
        # best es2s and best self._measuer_data is overwritten in the
        # loop above.  Rerun the lines below to save this again.
        es1s, es2s = self.get(dt_max)
        self.measure(es1s, es2s) # rerun to store correct self._measure_data
        self.last_dt_max = dt_max

        # Clean up, save old edge set.
        old_tstart = self.interval_low = self.tstart
        self.tstart = old_tstart + dt_max


        if self.old_es is None:
            # first round, comparing two expanding intervals.
            if self.merge_first:
                # Merge the first two intervals into a big one.
                self.old_es = es1s.union(es2s)
                self.tstart += dt_max  # double interval
                self.interval_high = self.tstart
                # Double dts, since our dt actually represents a
                # doubled interval.
                self._finder_data['dts'] = \
                           numpy.multiply(2, self._finder_data['dts'])
                self._finder_data['ts'] = \
                      numpy.subtract(self._finder_data['ts'], old_tstart) \
                        * 2+old_tstart
                if not self.quiet:
                    print('### merging first two intervals')

                self.old_n_events = self.evs.count_interval(old_tstart, self.tstart)
                return old_tstart, self.tstart
            else:
                # Old (normal) way of handling the first interval.
                self.old_es = es1s
                self.old_n_events = self.evs.count_interval(old_tstart, self.tstart)
                self.interval_high = self.tstart
                return old_tstart, self.tstart
        else:
            # Normal continuing process.
            self.old_es = es2s
            self.old_n_events = self.evs.count_interval(old_tstart, self.tstart)
            self.interval_high = self.tstart
            return old_tstart, self.tstart
    def find_uniform(self):
        """Alternative version of find that returns uniform intervals.

        This can not be set via an option, but must be set by manual
        code so far:
          dynsnap.SnapshotFinder.find = dynsnap.SnapshotFinder.find_uniform
          dynsnap.SnapshotFinder.dt_uniform = dt_uniform

        The only purpose is this method is to override *all* detection
        and return something for testing.
        """
        # Uniform initialization
        self.t_crit = False
        if self.tstart >= self.tstop:
            return None
        # Propagation
        old_tstart = self.tstart
        self._finder_data = dict(xs=[], ts=[], dts=[],
                                 measure_data=[])
        es1s, es2s = self.get(self.dt_uniform)
        x = self.measure(es1s, es2s)
        #self._finder_data['xs'].append(x)
        #self._finder_data['ts'].append(self.tstart+self.dt_uniform)
        #self._finder_data['dts'].append(self.dt_uniform)
        #self._finder_data['measure_data'].append(self._measure_data)
        self.found_x_max = x
        self.interval_low = old_tstart
        self.tstart = self.tstart + self.dt_uniform
        self.interval_high = self.tstart
        self.old_n_events = self.evs.count_interval(old_tstart, self.tstart)
        if self.old_es is None:
            self.old_es = es1s
        else:
            self.old_es = es2s
        return self.interval_low, self.interval_high

class Results(object):
    """This holds results and provides various analysis methods."""
    def __init__(self, finder, args=None):
        self.finder = finder
        self.args = args
        self.tlows = [ ]
        self.thighs = [ ]
        self.sims = [ ]
        self.finding_data = [ ]
        self.n_distinct = [ ]
        self.n_events = [ ]
        self.t_crit = [ ]

    def add(self, finder):
        """Record state, for used in plotting"""
        tlow  = finder.interval_low
        thigh = finder.interval_high
        self.tlows.append(tlow)
        self.thighs.append(thigh)
        self.sims.append(finder.found_x_max)
        self.n_distinct.append(len(finder.old_es))
        self.n_events.append(finder.old_n_events)
        # The following things don't always need to be stored
        self.finding_data.append((finder._finder_data['ts'],
                                  finder._finder_data['xs']))
        if finder.t_crit:
            self.t_crit.append(finder.interval_low)
    def plot_1(self, path, callback=None, **kwargs):
        """Plot of snapshot lengths and similarity.

        Save to path.[pdf,png]."""
        # If we have no data, don't do anything:
        if len(self.tlows) == 0:
            return
        try:
            import pcd.support.matplotlibutil as mplutil
            raise ImportError
        except ImportError:
            import mplutil
        fname = path + '.[pdf,png]'
        fig, extra = mplutil.get_axes(fname, figsize=(10, 10),
                                      ret_fig=True)
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax.set_xlabel('time (also snapshots intervals)')
        ax.set_ylabel('Snapshot length (t)')
        ax.set_xlabel('time')
        ax2.set_ylabel('Jaccard score (or measure)')

        points = [ ]
        for tlow, thigh in zip(self.tlows, self.thighs):
            points.append((tlow,  thigh-tlow))
            points.append((thigh, thigh-tlow))
        x, y = zip(*points)

        ax.set_xlim(self.tlows[0], self.thighs[-1])
        ax2.set_xlim(self.tlows[0], self.thighs[-1])

        ls = ax.plot(x, y, '-o')
        for tlow, thigh, (ts, xs) in zip(self.tlows, self.thighs, self.finding_data):
            ls = ax2.plot(ts, xs, '-')
            #ax.axvline(x=new_tstart, color=ls[0].get_color())
            if self.args.get('annotate_peaks', False):
                ax2.annotate(str(thigh), xy=(thigh, max(xs)))

        if callback:
            callback(locals())

        mplutil.save_axes(fig, extra)
    def plot_Jfinding(self, ax, convert_t=lambda t: t):
        """Plot the J finding at every interval"""
        xlow, xhigh = ax.get_xlim()
        for tlow, thigh, (ts, xs) in zip(self.tlows, self.thighs,
                                         self.finding_data):
            ls = ax.plot([convert_t(t) for t in ts], xs, '-')
            #ax.axvline(x=new_tstart, color=ls[0].get_color())
            if self.args.get('annotate_peaks', False):
                ax.annotate(str(convert_t(thigh)), xy=(convert_t(thigh), max(xs)))
        ax.set_xlim(xlow, xhigh)
        return ls

    def plot_density(self, ax, evs,
                     convert_t=lambda t: t,
                     style='-',
                     scale=1,
                     **kwargs):
        """Create local event density and plot it on an axis."""
        # Calculate local event density for the plot
        tlow = self.tlows[0]
        thigh = self.thighs[-1]
        data_size = thigh-tlow
        interval = data_size/1000.
        halfwidth = data_size/200
        tlow = math.floor(tlow/interval)*interval
        thigh = math.ceil(thigh/interval)*interval
        domain = numpy.arange(tlow, thigh, interval)
        #domain, densities = evs.event_density(domain=domain, halfwidth=halfwidth)
        domain, densities = evs.event_density(domain=domain, halfwidth=None,
                                              high=0, low=2*halfwidth)
        domain = [convert_t(x) for x in domain]
        densities = numpy.asarray(densities, dtype=float)
        densities = numpy.divide(densities, halfwidth*2)
        #print sum(densities[numpy.isnan(densities)==False])
        ls = ax.plot(domain, densities*scale, '-', color='#3F1F9E', zorder=0)
        return ls

    def plot_similarities(self, ax,
                          convert_t=lambda t: t,
                          style='g-'):
        # Plot similarities
        ts = self.tlows[:1] + self.thighs[:-1]
        sims = self.sims
        ls = ax.plot([convert_t(t) for t in ts], sims, style, color='#006F07')
        return ls
    def plot_intervals(self, ax,
                       convert_t=lambda t: t,
                       style='g-'):
        ts = self.thighs
        # Plot vertical lines for intervals
        lines = [ ]
        for thigh in ts:
            ls = ax.axvline(x=convert_t(thigh), color=(.21,.21,.21), linewidth=.5)
            lines.append(ls)
        return lines
    def plot_intervals_patches(self, ax,
                               convert_t=lambda t: t,
                               shift=0,
                               ylow=0,
                               yhigh=1):
        """Plot detected intervals as patches."""
        import matplotlib.patches
        import matplotlib.dates as mdates
        if isinstance(convert_t(0), datetime.datetime):
            def calc_width(high, low):
                low = convert_t(low)
                high = convert_t(high)
                return mdates.date2num(high) - mdates.date2num(low)
        else:
            def calc_width(high, low):
                return high-low
        #ylow, yhigh = ax.get_xlim()
        ts = self.thighs
        if len(ts)%2 == 1: # if odd
            ts = [self.tlows[0]] + ts
        # Plot vertical lines for intervals
        patches = [ ]
        for tA, tB in zip(ts[0::2], ts[1::2]):
            #p = ax.add_patch(matplotlib.patches.Rectangle(
            #          (convert_t(tA+shift), ylow),
            #          calc_width(tB, tA),
            #          yhigh-ylow,
            #          facecolor=(.81,.81,.81),
            #          edgecolor=(.51,.51,.51),
            #          zorder=-10))
            p = ax.axvspan(convert_t(tA+shift), convert_t(tB+shift),
                           facecolor=(.81,.81,.81),
                           edgecolor=(.51,.51,.51),
                           zorder=-10)
            patches.append(p)
        return patches
    def plot_actual(self, ax,
                    convert_t=lambda t: t,
                    style='o', color='red'):
        for e, label in major_events:
            #ax.axvline(x=e, color='red')
            ax.plot([convert_t(e)], 1, 'o', color='red')

    def plot_2(self, path, callback=None, evs=None, convert_t=lambda t: t, **kwargs):
        """Plot of similarity scores and local event density.

        Save to path.[pdf,png]."""
        if len(self.tlows) == 0:
            return
        try:
            import pcd.support.matplotlibutil as mplutil
            raise ImportError
        except ImportError:
            import mplutil
        fname = path + '.[pdf,png]'
        fig, extra = mplutil.get_axes(fname, figsize=(10, 5),
                                      ret_fig=True)
        ax = fig.add_subplot(1, 1, 1)
        ax2 = ax.twinx()
        ax.set_xlabel('time')
        ax.set_ylabel('Local event density')
        ax2.set_ylabel('Similarity score')

        ax.set_xlim(convert_t(self.tlows[0]), convert_t(self.thighs[-1]))
        #ax2.set_xlim(self.tlows[0], self.thighs[-1])

        # Calculate local event density for the plot
        tlow = self.tlows[0]
        thigh = self.thighs[-1]
        data_size = thigh-tlow
        interval = data_size/1000.
        halfwidth = data_size/100.
        tlow = math.floor(tlow/interval)*interval
        thigh = math.ceil(thigh/interval)*interval
        domain = numpy.arange(tlow, thigh, interval)
        domain, densities = evs.event_density(domain=domain, halfwidth=halfwidth)
        densities = numpy.asarray(densities, dtype=float)
        densities = numpy.divide(densities, halfwidth*2)
        #print sum(densities[numpy.isnan(densities)==False])

        # Transform domain into human-readable times, if wanted.
        domain = [convert_t(t) for t in domain ]

        # Plot local density.
        ls = ax.plot(domain, densities, '-')
        ax.set_yscale("log")
        adf = ax.xaxis.get_major_formatter()
        if hasattr(adf, 'scaled'):
            adf.scaled[1./(24*60)] = '%H:%M'  # set the < 1d scale to H:M
            adf.scaled[1./24] = '%H:%M'  # set the < 1d scale to H:M
            adf.scaled[1.0] = '%m-%d' # set the > 1d < 1m scale to Y-m-d
            adf.scaled[30.] = '%Y-%m' # set the > 1m < 1Y scale to Y-m
            adf.scaled[365.] = '%Y' # set the > 1y scale to Y

        # Plot similarities
        ts = self.thighs
        sims = self.sims
        ls = ax2.plot([convert_t(t) for t in ts], sims, 'g-')

        # Plot vertical lines for intervals
        for thigh in ts:
            ax.axvline(x=convert_t(thigh), color='k', linewidth=.5)

        if callback:
            callback(locals())

        mplutil.save_axes(fig, extra)
    def tf_idf(self, evs, n=10):
        """TF-IDF analysis of intervals.

        This functions calculates the term frequency-inverse document
        frequency of events within intervals.  This allows one to see
        the most characteristic events within each interval.

        Term frequency: fraction of term weights each term occupies
        within an interval.  This does consider weights always.

        Inverse document frequency: -log10(n/N), N=total intervals,
        n=intervals containing term.  Events occuring in all intervals
        are excluded from analysis.  This does not consider weights.

        The returned value is the product of both of these.  The top
        10 terms are returned.
        """
        # calculate document frequency for each term.
        dfs = collections.defaultdict(int)
        for tlow, thigh in zip(self.tlows, self.thighs):
            c = evs._execute("""SELECT DISTINCT e from %s WHERE ?<=t AND t<?"""%evs.table,
                             (tlow, thigh))
            for (e, ) in c:
                dfs[e] += 1
        # make the logarithmic DF
        for e in dfs:
            #if dfs[e] == float(len(self.tlows)):
            #    print dfs[e], float(len(self.tlows)), -log10(dfs[e]/float(len(self.tlows))), \
            #          evs.get_event_names(e)
            dfs[e] = -log10(dfs[e]/float(len(self.tlows)))
        dfs = dict(dfs)
        # For each interval, compute TFIDF
        return_data = [ ]
        for tlow, thigh in zip(self.tlows, self.thighs):
            total_terms = evs._execute("""SELECT sum(w) from %s WHERE ?<=t AND t<? """%evs.table,
                             (tlow, thigh)).fetchone()[0]

            c = evs._execute("""SELECT e, sum(w)/? from %s WHERE ?<=t AND t<? GROUP BY e"""%evs.table,
                             (total_terms, tlow, thigh))
            tfs = dict(c.fetchall())
            items = [  ]
            mostcommon = heapq.nlargest(10,
                                        ((tf*dfs[e], e) for e, tf in iteritems(tfs) if dfs[e]!=0),
                                        key=lambda x: x[0])
            if mostcommon:
                names = evs.get_event_names(tuple(zip(*mostcommon))[1])
            else:
                names = [ ]
            #print tlow, thigh
            #for (tfidf, e), name in zip(mostcommon, names):
            #    print "    %5.2f %d %s"%(tfidf, tfs[e], name)
            return_data.append((tfidf, name) for (tfidf, e), name in zip(mostcommon, names))
        return return_data
    def entropy(self, evs):
        """Calculate entropies in detected intervals.  Return entropies."""
        all_S = [ ]
        for tlow, thigh in zip(self.tlows, self.thighs):
            c = evs._execute("""SELECT count(*) from %s WHERE ?<=t AND t<? GROUP BY e"""%evs.table,
                             (tlow, thigh))
            # Could use scipy.stats.entropy, but not depending on
            # scipy yet.
            counts = [x for (x,) in c ]
            total = float(sum(counts))
            S = -sum(p/total*log(p/total) for p in counts if p>0 ) / log(2)
            all_S.append(S)
        return all_S
    def plot_entropy(self, evs, ax, convert_t=lambda t: t):
        all_S = self.entropy(evs)
        points = sum( ([(convert_t(tlow), S), (convert_t(thigh),S) ]
                        for (tlow, thigh, S)
                        in zip(self.tlows, self.thighs, all_S)),
                     []
                    )
        xs, ys = zip(*points)
        ls = ax.plot(xs, ys, '-', c='#E6C416')
        print('xxx'*50)
        return ls



# Main argument parser.  Left here becaues it is used in multiple
# functions.
parser = argparse.ArgumentParser()
parser.add_argument("input", help="benchmark model to simulate",)
parser.add_argument("output", help="Output prefix", nargs='?')
parser.add_argument("--measure", default=SnapshotFinder.measure,
                    help="Similarity measure (jacc, cosine, cosine_uw) "
                    "(default %(default)s)")
parser.add_argument("--cache", action='store_true',
                    help="Cache input for efficiency")
parser.add_argument("--regen", action='store_true',
                    help="Recreate temporal event cache")
parser.add_argument("--unordered", action='store_true',
                    help="Event columns on the line are unordered")
parser.add_argument("--grouped", action='store_true',
                    help="Each line contains different space-separated "
                    "events.")
parser.add_argument("--stats", action='store_true',
                    help="Don't do segmentation, just print stats on the data.")
parser.add_argument("--merge-first", action='store_true', dest="merge_first",
                    default=SnapshotFinder.merge_first,
                    help="Merge the first two intervals (default %(default)s).")

parser.add_argument("-t",  type=int, default=0,
                    help="Time column")
parser.add_argument("-w", type=int, default=None,
                    help="Weight column")
parser.add_argument("--datacols", default="",
                    help="Columns containing the data"
                         " (comma separated list)")
parser.add_argument("-p", "--plot", action='store_true',
                    help="Plot also?")
parser.add_argument("--plotstyle", default='2',
                    help="Plot style, '1' or '2'.")
parser.add_argument("-i", "--interact", action='store_true',
                    help="Interact with results in IPython after calculation")

parser.add_argument("--tformat", help="Parse time in this format.  Options: 'unixtime'")
parser.add_argument("--tstart", type=float, help="Time to begin analysis.")
parser.add_argument("--tstop", type=float, help="Time to end analysis.")
parser.add_argument("--dtmode", default=SnapshotFinder.dtmode,
                    help="dt search pattern (linear, log, event) "
                         "(default: %(default)s)")

group = parser.add_argument_group("Time scale related options")
group.add_argument("--peakfinder", default=SnapshotFinder.peakfinder,
                    help="How to select peak of Jaccard similarity. "
                         "(shortest, longest, greedy) "
                         "(default=%(default)s)")
group.add_argument("--dt-pastpeak-factor", type=float, metavar='FACTOR',
                   help="Factor by which to search past the maximum for a better maximum "
                   "(not greedy peakfinder) (default %(default)s)")
group.add_argument("--dt-peak-factor", type=float, metavar='FACTOR',
                   default=SnapshotFinder.dt_peak_factor,
                   help="Immediately stop seaching if similarity drops to "
                   "this fracion of the peak.  Range: [0,1). "
                   "(default=%(default)s)"
                   )
group.add_argument("--dt-search-min", default=SnapshotFinder.dt_search_min, metavar='DT',
                   type=float, help="Minimum amount to seach on each "
                                    "step (default %(default)s.")
group.add_argument("--dt-pastpeak-min", type=float, metavar='DT',
                   default=SnapshotFinder.dt_pastpeak_min,
                   help="Minimum amount to search past each peak (not greedy) (default %(default)s)")
group.add_argument("--dt-pastpeak-max", type=float, metavar='DT',
                   default=SnapshotFinder.dt_pastpeak_max,
                   help="Maximum amount to search past each peak (not greedy) (default %(default)s)")

group = parser.add_argument_group("Linear time options (must specify --dtmode=linear)")
group.add_argument("--dtstep", type=float, default=SnapshotFinder.dt_step,
                   help="step size for dt scanning. (default=%(default)s)")
group.add_argument("--dtmin", type=float, help="(default=DTSTEP)")
group.add_argument("--dtmax", type=float, help="(default=1000*DTSTEP)")

group = parser.add_argument_group("Logarithmic time options (with --dtmode=log)")
group.add_argument("--log-dtmin", type=float,)
group.add_argument("--log-dtmax", type=float,)


def main(argv=sys.argv[1:], return_output=True, evs=None,
         convert_t=None, outsuffix=None):
    """Frontend to all detection.

    Despite being named main() and being used in the command line, this
    can be used to encapsulate all detection in your code.  Simply
    create the command line arguments that you want (argv), pass in an
    Events object (if your arguments do not read it in) (evs), and you
    get a Result object out (and any other output files, as specified by
    the command line options).

    Unfortunately this is kind of confusing, but looking it (search for
    all uses of the variable "finder" you can see the steps needed to
    run SnapshotFinder.  Perhaps sometime these would be added to
    SnapshotFinder itself.
    """

    args = parser.parse_args(args=argv)
    #print args
    if outsuffix and args.output:
        args.output = args.output + outsuffix

    if evs is None:
        evs = load_events(args.input, col_time=args.t,
                          col_weight=args.w, cache=args.cache,
                          cols_data=args.datacols,
                          regen=args.regen,
                          unordered=args.unordered,
                          grouped=args.grouped)
        print("# file loaded:", args.input)

    finder = SnapshotFinder(evs, tstart=args.tstart, tstop=args.tstop,
                            args=args,
                            weighted=(args.w is not None),
                            dtmode=args.dtmode,
                            peakfinder=args.peakfinder,
                            measure=args.measure,
                            merge_first=args.merge_first,

                            dt_pastpeak_factor = args.dt_pastpeak_factor,
                            dt_peak_factor = args.dt_peak_factor,
                            dt_pastpeak_min = args.dt_pastpeak_min,
                            dt_pastpeak_max = args.dt_pastpeak_max,
                            dt_search_min = args.dt_search_min,

                            # linear options
                            dt_min   = args.dtmin,
                            dt_max   = args.dtmax,
                            dt_step  = args.dtstep,

                            # logarithmic search
                            log_dt_min = args.log_dtmin,
                            log_dt_max = args.log_dtmax,
                            )


    # Time format specification (for output files and stdout)
    format_t = lambda x: x   # null formatter
    if convert_t is not None:
        pass
        format_t = lambda t: \
                   convert_t(t).strftime('%Y-%m-%d_%H:%M:%S')
    elif args.tformat == 'unixtime':
        # Formatter for unix time (seconds since 1970-01-01 00:00 UTC)
        format_t = lambda t: \
                   datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d_%H:%M:%S')
        convert_t = lambda t: \
                   datetime.datetime.fromtimestamp(t)
    elif args.tformat:
        raise ValueError('Unknown --tformat: %s'%args.tformat)
    else:
        convert_t = lambda x: x  # conversion to datetime object for plotting, if applicable
    format_t_log = format_t

    # Print basic stats and exit, if requested.
    if args.stats:
        for a, b in evs.stats(convert_t=convert_t, tstart=args.tstart, tstop=args.tstop):
            print(a, b)
        exit(0)


    print("# Total time range:", format_t(evs.t_min()), format_t(evs.t_max()))
    #evs.dump()

    if args.interact:
        return_output = True
    if return_output:
        output = [ ]
    if args.output:
        # Make directory for output, if it doesn't exist already.
        dir_ = os.path.dirname(args.output)
        if dir_:
            if not os.path.isdir(dir_):
                os.makedirs(dir_)
        #
        fout_thresh = open(args.output+'.out.txt', 'w')
        fout_full = open(args.output+'.out.J.txt', 'w')
        print('#tlow thigh dt sim len(old_es) measure_data', file=fout_thresh)
        print('#t sim dt measure_data', file=fout_full)
    if return_output or args.plot:
        results = Results(finder, args=args.__dict__)

    time_last_plot = time.time()

    print('# Columns: tlow thigh dt sim number_of_events')
    try:
      while True:
        x = finder.find()
        if x is None:
            break
        tlow  = x[0]
        thigh = x[1]
        dt = thigh-tlow
        val = finder.found_x_max
        print(format_t(tlow), format_t(thigh), dt, val, len(finder.old_es))
        # Write and record informtion
        if return_output:
            output.append(ResultsRow(tlow, thigh, dt, val,
                                     finder._measure_data, finder._finder_data))
        if args.output:
            print(format_t_log(tlow), format_t_log(thigh),
                  dt, val, len(finder.old_es),
                  finder._measure_data,
                  file=fout_thresh)
            print('# t1=%s t2=%s dt=%s'%(format_t_log(tlow),
                                         format_t_log(thigh),
                                         thigh-tlow),
                  file=fout_full)
            print('# sim=%s'%val, file=fout_full)
            print('# len(old_es)=%s'%len(finder.old_es), file=fout_full)
            #print >> fout, '# len(old_es)=%s'%len(finder.old_es)
            for i, t in enumerate(finder._finder_data['ts']):
                print(finder._finder_data['dts'][i],
                      format_t_log(t),
                      finder._finder_data['xs'][i],
                      finder._finder_data['measure_data'][i],
                      file=fout_full)
            print(file=fout_full)
            fout_full.flush()

        if return_output or args.plot:
            results.add(finder)
            # Plot a checkpoint if we are taking a long time.
            if args.plot and time.time() > time_last_plot + 300:
                getattr(results, 'plot_'+args.plotstyle)(args.output, evs=evs, convert_t=convert_t)
                time_last_plot = time.time()
    except KeyboardInterrupt:
        # finalize plotting then re-raise.
        if args.plot:
            getattr(results, 'plot_'+args.plotstyle)(args.output, evs=evs, convert_t=convert_t)
        raise

    if args.plot:
        getattr(results, 'plot_'+args.plotstyle)(args.output, evs=evs, convert_t=convert_t)
    # print TFIDF data:
    if args.output and args.plot:
        tfidfs = results.tf_idf(evs, n=10)
        fout_tfidf = open(args.output+'.out.tfidf.txt', 'w')
        print('#tlow thigh dt tfidf term', file=fout_tfidf)
        for (tlow, thigh, terms) in zip(results.tlows, results.thighs, tfidfs):
            print(format_t_log(tlow), format_t_log(thigh), thigh-tlow,
                  '-', '-',
                  file=fout_tfidf)
            for x, name in terms:
                print('-', '-', '-', x, name.encode('utf-8'), file=fout_tfidf)

    if args.interact:
        import IPython
        IPython.embed()
    if return_output:
        return output, dict(finder=finder,
                            results=results, convert_t=convert_t)

def run_dual(argv=sys.argv[1:], return_output=True, evs=None,
             ax1=None, ax2=None,
             convert_t=None):
    """This was a shortcut method to run for both weighted and
    unweighted analyzes, and save to two matplotlib axes."""

    results_uw = main(argv=argv,                return_output=True, evs=evs, convert_t=convert_t, outsuffix='_uw')
    results_w = None
    if ax2:
        results_w  = main(argv=argv + ['-w', '-1'], return_output=True, evs=evs, convert_t=convert_t, outsuffix='_w')
    else:
        ax2 = None

    if not ax1:
        raise ValueError('Output to file not implemented yet.')

    results_uw[1]['results'].plot_similarities(ax1,
                                    convert_t=results_uw[1]['convert_t'])
    if ax2:
        results_w[1]['results'].plot_similarities(ax2,
                                    convert_t=results_w[1]['convert_t'])

    #results_uw[1]['results'].plot_intervals(ax1,
    #                                convert_t=results_uw[1]['convert_t'])
    results_uw[1]['results'].plot_intervals_patches(ax1,
                                    convert_t=results_uw[1]['convert_t'])
    if ax2:
        results_w[1]['results'].plot_intervals_patches(ax2,
                                    convert_t=results_w[1]['convert_t'])

    return results_uw, results_w


if __name__ == '__main__':
    main(argv=sys.argv[1:], return_output=False)
