# Richard Darst, November 2014

import argparse
import datetime
from math import log10
import sqlite3

from events import Events, load_events

class _LenProxy(object):
    def __init__(self, l):   self.l = l
    def __len__(self): return self.l
class WeightedSet(object):
    def __init__(self, it):
        data = self._data = { }
        for x, w in it:
            if x not in data:    data[x] = w
            else:                data[x] += w
    @classmethod
    def _from_data(cls, data):
        self = cls([])
        self._data = data
        return self
    def update(self, it):
        data = self._data
        for x, w in it:
            if x not in data:    data[x] = w
            else:                data[x] += w
    def __len__(self):
        return len(self._data)
    def __and__(self, other):
        """Set intersection"""
        if len(self) <= len(other):
            A, B = self._data, other._data
        else:
            A, B = other._data, self._data
        # A is the smaller set
        intersection = 0.0
        for x, w in A.iteritems():
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
        for x, w in A.iteritems():
            union += max(0,   A[x] - B.get(x, 0))
        return _LenProxy(union)
    def union(self, other):
        if len(self) <= len(other):
            A, B = self._data, other._data
        else:
            A, B = other._data, self._data
        data = dict(B)
        # A is the smaller set
        for x, w in A.iteritems():
            if x not in data:    data[x] = w
            else:                data[x] += w
        return self._from_data(data)

def test_weightedset():
    from nose.tools import assert_equal
    A = WeightedSet([('a',1), ('b',2),])
    B = WeightedSet([('b',1), ('c',2),])
    C = WeightedSet([('a',1), ('c',1), ('d',3)])
    assert_equal(len(A & B), 1)
    assert_equal(len(A | B), 5)
    assert_equal(len(A & C), 1)
    assert_equal(len(A | C), 7)





#import networkx as nx
#import pcd.cmtycmp
#import pcd.support.algorithms as algs
import numpy

class SnapshotFinder(object):
    old_es = None   # None on first round
    old_incremental_es = None
    dt_min = 1
    dt_max = 1000
    dt_step = 1
    dt_extra = 50
    log_dt_max = None

    def __init__(self, evs, tstart=None, tstop=None, weighted=False,
                 dtmode='linear', maxmode='shortest',
                 args={},
                 dt_min=1, dt_max=1000, dt_step=1, dt_extra=50,
                 log_dt_max=None,
                 ):
        self.evs = evs
        if isinstance(args, argparse.Namespace):
            args = args.__dict__
        self.args = args
        self.weighted = weighted

        if tstart is not None:    self.tstart = tstart
        else:                     self.tstart = evs.t_min()
        if tstop is not None:     self.tstop  = tstop
        else:                     self.tstop  = evs.t_max()

        if dtmode == 'linear':    self.iter_all_dts = self.iter_all_dts_linear
        elif dtmode == 'log':     self.iter_all_dts = self.iter_all_dts_log
        else:                     raise ValueError("Unknown dtmode: %s"%dtmode)

        if   maxmode == 'shortest':  self.pick_best_dt = self.pick_best_dt_shortest
        elif maxmode == 'longest':   self.pick_best_dt = self.pick_best_dt_longest
        elif maxmode == 'greedy':    self.pick_best_dt = self.pick_best_dt_greedy
        else:                        raise ValueError("Unknown maxmode: %s"%maxmode)

        locals_ = locals()
        for name in ('dt_min', 'dt_max', 'dt_step', 'dt_extra', 'log_dt_max'):
            if locals_[name] is not None:
                setattr(self, name, locals_[name])
        #self.dt_min     = dt_min
        #self.dt_max     = dt_max
        #self.dt_step    = dt_step
        #self.dt_extra   = dt_extra
        #self.log_dt_max = log_dt_max


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
        """Get a first interval: two intervals"""
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
    def measure_esjacc(self, es1s, es2s):
        union = len(es1s | es2s)
        if union == 0:
            x = float('nan')
        else:
            intersect = len(es1s & es2s)
            x = intersect / float(union)
            self._measure_data = (intersect, union, len(es1s), len(es2s))
        return x
    def measure_nmi(self, es1s, es2s):
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
    measure = measure_esjacc


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
    def iter_all_dts_log(self):
        dt = 1
        log_scale = -1  # 0 = 1,2,3,..10,20,..100,200
                        #-1 = 1,2,..10,11...100,110,333
        while True:
            #print dt
            yield dt
            dt += int(10**( max(0, int(log10(dt))+log_scale)  ))
            if self.log_dt_max and dt > self.log_dt_max: break
            if self.tstart + dt > self.tstop: break

    iter_all_dts = iter_all_dts_linear
    class StopSearch(BaseException):
        def __init__(self, i_max, *args, **kwargs):
            self.i_max = i_max
            return super(self.__class__, self).__init__(*args, **kwargs)

    def pick_best_dt_shortest(self, dt, dts, xs):
        i_max = numpy.argmax(xs)

        dt_extra_ = self.dt_extra
        if not dt_extra_:
            dt_extra_ = 2*dts[i_max]
        #print 'dt_extra:', dt_extra_, i_max, dts[i_max], self.dt_extra

        if dt > dts[i_max] + dt_extra_:
            raise self.StopSearch(i_max)
        return i_max
    def pick_best_dt_longest(self, dt, dts, xs):
        xs_reversed = xs[::-1]
        i_max = numpy.argmax(xs_reversed)
        x_max = len(xs)-1-i_max   # undo the reversal
        raise NotImplementedError("this has bugs") # i_max not updated

        dt_extra_ = self.dt_extra
        if not dt_extra_:
            dt_extra_ = 2*dt

        if dt > dts[i_max] + dt_extra_:
            raise self.StopSearch(i_max)
        return i_max
    def pick_best_dt_greedy(self, dt, dts, xs):
        if len(xs) > 2 and xs[-2] > xs[-1]:  # if we have decreased on the last step
            raise self.StopSearch(len(xs)-2)
        # break condition when exceeding scan time
        dt_extra_ = self.dt_extra
        if self.tstart + dt > self.tstop:
            raise self.StopSearch(len(xs)-1)


        return len(xs) - 1
    pick_best_dt = pick_best_dt_shortest

    # This is the core function that does a search.
    def find(self):
        """Core snapshot finding method.

        Returns (low, high) values."""
        dts = [ ]
        xs = [ ]
        self._finder_data = dict(xs=[], ts=[], dts=[],
                                 measure_data=[])
        i_max = None
        try:
          for i, dt in enumerate(self.iter_all_dts()):
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
        if (xs[i_max] == xs[-1]
            #and xs[i_max] == xs[-1]
            and xs[i_max] == 0
            and self.old_es is not None):
            # At this point, we have had an extreme event and we
            # should restart from zero.
            self.old_es = None # Remove old interval.  We need a fresh
                               # restart.
            return self.find()

        #print xs
        dt_max = self.found_dt_max = dts[i_max]
        x_max  = self.found_x_max  = xs[i_max]
        # best es2s and best self._measuer_data is overwritten in the
        # loop above.  Rerun the lines below to save this again.
        es1s, es2s = self.get(dt_max)
        self.measure(es1s, es2s) # rerun to store correct self._measure_data

        # Clean up, save old edge set.
        old_tstart = self.interval_low = self.tstart
        self.tstart = old_tstart + dt_max


        if self.old_es is None:
            # first round, comparing two expanding intervals.
            if self.args.get('merge_first', True):
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
                print 'doubling'

                return old_tstart, self.tstart
            else:
                # Old (normal) way of handling the first interval.
                self.old_es = es1s
                self.interval_high = self.tstart
                return old_tstart, self.tstart
        else:
            # Normal continuing process.
            self.old_es = es2s
            self.interval_high = self.tstart
            return old_tstart, self.tstart

        #print "  %4d %3d %3d %s"%(self.tstart, dt_max, i_max, len(dts))



class Plotter(object):
    def __init__(self, finder, args=None):
        self.finder = finder
        self.args = args
        self.points = [ ]
        self.finding_data = [ ]
    def add(self, finder):
        """Record state, for used in plotting"""
        tlow  = finder.interval_low
        thigh = finder.interval_high
        self.points.append((tlow,  thigh-tlow))
        self.points.append((thigh, thigh-tlow))
        self.finding_data.append((finder._finder_data['ts'],
                               finder._finder_data['xs'],
                               finder.interval_low,
                               finder.interval_high))
    def plot(self, path, callback=None):
        """Do plotting.  Save to path.[pdf,png]"""
        # If we have no data, don't do anything:
        if len(self.points) == 0:
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

        x, y = zip(*self.points)

        ax.set_xlim(x[0], x[-1])
        ax2.set_xlim(x[0], x[-1])

        ls = ax.plot(x, y, '-o')
        for ts, xs, tlow, thigh in self.finding_data:
            ls = ax2.plot(ts, xs, '-')
            #ax.axvline(x=new_tstart, color=ls[0].get_color())
            if self.args.get('annotate_peaks', False):
                ax2.annotate(str(thigh), xy=(thigh, max(xs)))

        if callback:
            callback(locals())

        mplutil.save_axes(fig, extra)



if __name__ == '__main__':
    from itertools import product
    import math
    import numpy
    import os
    import random
    import time

    import networkx

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="benchmark model to simulate",)
    parser.add_argument("output", help="Output prefix", nargs='?')
    parser.add_argument("--cache", action='store_true',
                        help="Cache input for efficiency")
    parser.add_argument("--regen", action='store_true',
                        help="Recreate temporal event cache")
    parser.add_argument("--unordered", action='store_true',
                        help="Event columns on the line are unordered")
    parser.add_argument("--grouped", action='store_true',
                        help="Each line contains different space-separated "
                        "events.")
    parser.add_argument("--dont-merge-first", action='store_false',
                        default=True, dest="merge_first",
                        help="Each line contains different space-separated "
                        "events.")

    parser.add_argument("-t",  type=int, default=0,
                        help="Time column")
    parser.add_argument("-w", type=int, default=None,
                        help="Weight column")
    parser.add_argument("-p", "--plot", action='store_true',
                        help="Plot also?")

    parser.add_argument("--tformat")
    parser.add_argument("--tstart", type=float, help="Time to begin analysis.")
    parser.add_argument("--tstop", type=float, help="Time to end analysis.")
    parser.add_argument("--dtmode", default='linear')
    parser.add_argument("--maxmode", default='shortest')

    group = parser.add_argument_group("Linear time options")
    group.add_argument("--dtmin", type=float,)
    group.add_argument("--dtmax", type=float,)
    group.add_argument("--dtstep", type=float,)
    group.add_argument("--dtextra", type=float, default=None)

    group = parser.add_argument_group("Logarithmic time options")
    group.add_argument("--log-dtmax", type=float,)

    args = parser.parse_args()
    #print args

    evs = load_events(args.input, col_time=args.t,
                      col_weight=args.w, cache=args.cache,
                      regen=args.regen,
                      unordered=args.unordered,
                      grouped=args.grouped)
    print "file loaded"

    finder = SnapshotFinder(evs, tstart=args.tstart, tstop=args.tstop,
                            args=args,
                            weighted=bool(args.w),
                            dtmode=args.dtmode,
                            maxmode=args.maxmode,

                            # linear options
                            dt_min   = args.dtmin,
                            dt_max   = args.dtmax,
                            dt_extra = args.dtextra,
                            dt_step  = args.dtstep,

                            # logarithmic search
                            log_dt_max = args.log_dtmax,
                            )


    # Time format specification (for output files and stdout)
    format_t = lambda x: x   # null formatter
    if args.tformat == 'unixtime':
        # Formatter for unix time (seconds since 1970-01-01 00:00 UTC)
        format_t = lambda t: \
                   datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d_%H:%M:%S')
    format_t_log = format_t

    print format_t(evs.t_min()), format_t(evs.t_max())
    #evs.dump()

    points = [ ]
    finding_data = [ ]
    if args.output:
        # Make directory for output, if it doesn't exist already.
        dir_ = os.path.dirname(args.output)
        if dir_:
            if not os.path.isdir(dir_):
                os.makedirs(dir_)
        #
        fout_thresh = open(args.output+'.out.txt', 'w')
        fout_full = open(args.output+'.out.J.txt', 'w')
        print >> fout_thresh, '#tlow thigh dt val len(old_es) measure_data'
        print >> fout_full, '#t val dt measure_data'
    if args.plot:
        plotter = Plotter(finder, args=args.__dict__)


    time_last_plot = time.time()

    try:
      while True:
        x = finder.find()
        if x is None:
            break
        tlow  = x[0]
        thigh = x[1]
        dt = thigh-tlow
        val = finder.found_x_max
        print format_t(tlow), format_t(thigh), val, dt, len(finder.old_es)
        # Write and record informtion
        if args.output:
            print >> fout_thresh, format_t_log(tlow), format_t_log(thigh), \
                                  dt, val, len(finder.old_es), \
                  finder._measure_data
            print >> fout_full, '# t1=%s t2=%s dt=%s'%(format_t_log(tlow), format_t_log(thigh),
                                                       thigh-tlow)
            print >> fout_full, '# J=%s'%val
            print >> fout_full, '# len(old_es)=%s'%len(finder.old_es)
            #print >> fout, '# len(old_es)=%s'%len(finder.old_es)
            for i, t in enumerate(finder._finder_data['ts']):
                print >> fout_full, finder._finder_data['dts'][i], \
                                    format_t_log(t), \
                                    finder._finder_data['xs'][i], \
                                    finder._finder_data['measure_data'][i]
            print >> fout_full
            fout_full.flush()

        if args.plot:
            plotter.add(finder)
            # Plot a checkpoint if we are taking a long time.
            if time.time() > time_last_plot + 300:
                plotter.plot(args.output)
                time_last_plot = time.time()
    except KeyboardInterrupt:
        # finalize plotting then re-raise.
        if args.plot:
            plotter.plot(args.output)
        raise

    if args.plot:
        plotter.plot(args.output)

