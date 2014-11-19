# Richard Darst, November 2014

import sqlite3

class _LenProxy(object):
    def __init__(self, l):   self.l = l
    def __len__(self): return self.l
class WeightedSet(object):
    def __init__(self, it):
        data = self._data = { }
        for x, w in it:
            if x not in data:    data[x] = w
            else:                data[x] += w
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
        """Set intersection"""
        if len(self) <= len(other):
            A, B = self._data, other._data
        else:
            A, B = other._data, self._data
        # A is the smaller set
        union = sum(B.itervalues())
        for x, w in A.iteritems():
            union += max(0,   A[x] - B.get(x, 0))
        return _LenProxy(union)
def test_weightedset():
    from nose.tools import assert_equal
    A = WeightedSet([('a',1), ('b',2),])
    B = WeightedSet([('b',1), ('c',2),])
    C = WeightedSet([('a',1), ('c',1), ('d',3)])
    assert_equal(len(A & B), 1)
    assert_equal(len(A | B), 5)
    assert_equal(len(A & C), 1)
    assert_equal(len(A | C), 7)

class Events(object):
    def __init__(self, fname=':memory:'):
        self.conn = sqlite3.connect(fname)
        c = self.conn.cursor()

        c.execute('''CREATE TABLE if not exists event
                     (t int not null, e int not null, w real)''')
        c.execute('''CREATE INDEX if not exists index_event_t ON event (t)''')
        #c.execute('''CREATE INDEX if not exists index_event_e ON event (e)''')
        c.close()
    def add_event(self, t, e, w=1.0):
        c = self.conn.cursor()
        c.execute("INSERT INTO event VALUES (?, ?, ?)", (t, e, w))
        self.conn.commit()
    def add_events(self, it):
        c = self.conn.cursor()
        c.executemany("INSERT INTO event VALUES (?, ?, ?)", it)
        self.conn.commit()
    def t_min(self):
        c = self.conn.cursor()
        c.execute("SELECT min(t) from event")
        return c.fetchone()[0]
    def t_max(self):
        c = self.conn.cursor()
        c.execute("SELECT max(t) from event")
        return c.fetchone()[0]


    def __len__(self):
        c = self.conn.cursor()
        c.execute("SELECT count(*) from event")
        return c.fetchone()[0]

    def __getitem__(self, interval):
        assert isinstance(interval, slice)
        assert interval.step is None
        assert interval.start is not None, "Must specify interval start"
        c = self.conn.cursor()
        if interval.stop is not None:
            # Actual slice.
            c.execute('''select t, e, w from event where ? <= t AND t < ?''',
                      (interval.start, interval.stop, ))
            return c
        else:
            return _EventsListSubset(self, interval.start)

class _EventListSubset(object):
    def __init__(self, events, t_start):
        self.events = events
        self.t_start = t_start
    def __getitem__(self, interval):
        assert isinstance(interval, slice)
        assert interval.start is None
        assert interval.stop is not None

        c.execute('''select t, e, w from event where ? <= t AND t < ?''',
                  (self.t_start, interval.stop, ))
        return c




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
    weighted = False
    def __init__(self, evs):
        self.evs = evs
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

    def get_first(self, dt):
        cursor = self.evs[self.tstart: self.tstart+dt]
        es1 = self._set_make(cursor)
        cursor = self.evs[self.tstart+dt: self.tstart+2*dt]
        es2 = self._set_make(cursor)
        return es1, es2
    def get_succesive(self, dt):
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
        if self.old_es == None:
            return self.get_first(dt)
        else:
            return self.old_es, self.get_succesive(dt)

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

    def find(self):
        all_dts = numpy.arange(self.dt_min, self.dt_max+self.dt_step,
                               self.dt_step)
        # sqlite3 can't handle numpy.int64, convert all to floats.
        all_dts = [ eval(repr(x)) for x in all_dts ]

        dts = [ ]
        xs = [ ]
        self._finder_data = dict(xs=[], ts=[], dts=[],
                                 measure_data=[])
        es1max = es2max = None
        i_max = None
        for i, dt in enumerate(all_dts):
            es1s, es2s = self.get(dt)
            if len(es1s) == 0 or len(es2s) == 0:
                continue
            #es1s = set(es1)
            #es2s = set(es2)
            x = self.measure(es1s, es2s)
            if x == 0:
                continue
            #print dt, len(es1s), len(es2s), x

            self._finder_data['dts'].append(dt)
            self._finder_data['ts'].append(self.tstart+dt)
            self._finder_data['xs'].append(x)
            self._finder_data['measure_data'].append(self._measure_data)
            dts.append(dt)
            xs.append(x)

            i_max = numpy.argmax(xs)
            #if i_max == len(dts)-1:
            #    es1max = set(es1s)
            #    es2max = set(es2s)
            if dt > dts[i_max]+self.dt_extra:
                break
        if i_max is None:
            return None

        #print xs
        dt_max = self.found_dt_max = dts[i_max]
        x_max  = self.found_x_max  = xs[i_max]
        es1s, es2s = self.get(dt_max)  # to save correct es2s
        self.measure(es1s, es2s) # rerun to store correct self._measure_data

        tstart = self.tstart
        self.tstart = self.tstart + dt_max
        if self.old_es is None:
            # first round
            self.old_es = es1s
            return tstart, self.tstart
        else:
            self.old_es = es2s
            return tstart, self.tstart

        #print "  %4d %3d %3d %s"%(self.tstart, dt_max, i_max, len(dts))

import ast
import os
def load_events(fname, col_time=0, col_weight=None, cache=False, regen=False,
                unordered=False):
    events = { }
    def _iter():
        f = open(fname)
        for line in f:
            #line = line.split('#', 1)[0]
            line = line.strip()
            if line.startswith('#'): continue
            line = line.split()
            if not line: continue
            t = ast.literal_eval(line.pop(col_time))
            col_weight2 = col_weight # modified in this scope so needs
                                     # local copy
            if col_weight is not None and col_weight != -1:
                assert col_weight != col_time, ("weight column specified "
                                                "same as time column.")
                if col_weight > col_time:
                    # We removed one column, need to adjust.
                    col_weight2 -= 1
                w = ast.literal_eval(line.pop(col_weight2))
            else:
                w = 1.0
            if unordered:
                e = frozenset(line)
            else:
                e = ' '.join(line)
            if e in events:
                i = events[e]
            else:
                i = len(events)
                events[e] = i
            yield t, i, w
    if cache:
        cache_fname = fname + '.cache'
        if regen:
            # remove existing cache if it exists.
            if os.path.exists(cache_fname):
                os.unlink(cache_fname)
        if os.path.exists(cache_fname):
            ev = Events(cache_fname)
            return ev
    else:
        cache_fname = ':memory:'
    ev = Events(cache_fname)
    ev.add_events(_iter())
    return ev


if __name__ == '__main__':
    from itertools import product
    import math
    import numpy
    import random

    import networkx

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="benchmark model to simulate",)
    parser.add_argument("output", help="Output prefix", nargs='?')
    parser.add_argument("--cache", action='store_true',
                        help="Cache input for efficiency")
    parser.add_argument("--regen", action='store_true',
                        help="Recreate temporal event cache")
    parser.add_argument("--unordered", action='store_true',
                        help="Event columns on the line are unordered")

    parser.add_argument("-t",  type=int, default=0,
                        help="Time column")
    parser.add_argument("-w", type=int, default=None,
                        help="Weight column")
    parser.add_argument("-p", "--plot", action='store_true',
                        help="Plot also?")

    parser.add_argument("--tstart", type=float,)
    parser.add_argument("--dtmin", type=float,)
    parser.add_argument("--dtmax", type=float,)
    parser.add_argument("--dtstep", type=float,)
    parser.add_argument("--dtextra", type=float,)

    args = parser.parse_args()
    #print args

    evs = load_events(args.input, col_time=args.t,
                      col_weight=args.w, cache=args.cache,
                      regen=args.regen,
                      unordered=args.unordered)
    print "file loaded"

    finder = SnapshotFinder(evs)
    # Find tstart
    if args.tstart:
        finder.tstart = 1000
    else:
        finder.tstart = evs.t_min()
    if args.w is not None:
        finder.weighted = True
    #
    if args.dtextra is not None: finder.dt_extra = args.dtextra
    if args.dtmin   is not None: finder.dt_min   = args.dtmin
    if args.dtmax   is not None: finder.dt_max   = args.dtmax
    if args.dtstep  is not None: finder.dt_step  = args.dtstep


    points = [ ]
    finding_data = [ ]
    if args.output:
        fout_thresh = open(args.output+'.out.txt', 'w')
        fout_full = open(args.output+'.out.J.txt', 'w')
        print >> fout_thresh, '#tlow thigh dt val len(old_es) measure_data'
        print >> fout_full, '#t val dt measure_data'
    while True:
        x = finder.find()
        if x is None:
            break
        tlow  = x[0]
        thigh = x[1]
        dt = thigh-tlow
        val = finder.found_x_max
        print tlow, thigh, val, dt
        finding_data.append((finder._finder_data['ts'],
                             finder._finder_data['xs'],
                             finder.tstart))
        points.append((tlow,  thigh-tlow))
        points.append((thigh, thigh-tlow))
        # Write and record informtion
        if args.output:
            print >> fout_thresh, tlow, thigh, dt, val, len(finder.old_es), \
                  finder._measure_data
            print >> fout_full, '# t1=%s t2=%s dt=%s'%(tlow, thigh, thigh-tlow)
            print >> fout_full, '# J=%s'%val
            print >> fout_full, '# len(old_es)=%s'%len(finder.old_es)
            #print >> fout, '# len(old_es)=%s'%len(finder.old_es)
            for i, t in enumerate(finder._finder_data['ts']):
                print >> fout_full, t, \
                                    finder._finder_data['xs'][i], \
                                    finder._finder_data['dts'][i], \
                                    finder._finder_data['measure_data'][i]
            print >> fout_full
            fout_full.flush()



    if args.plot:
        try:
            import pcd.support.matplotlibutil as mplutil
            raise ImportError
        except ImportError:
            import mplutil
        fname = args.output+'.[pdf,png]'
        fig, extra = mplutil.get_axes(fname, figsize=(10, 10),
                                      ret_fig=True)
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax.set_xlabel('time (also snapshots intervals)')
        ax.set_ylabel('Snapshot length (t)')
        ax.set_xlabel('time')
        ax2.set_ylabel('Jaccard score (or measure)')
        ax.set_xlim(evs.t_min(), evs.t_max())
        ax2.set_xlim(evs.t_min(), evs.t_max())

        x, y = zip(*points)
        ls = ax.plot(x, y, '-o')
        for ts, xs, new_tstart in finding_data:
            ls = ax2.plot(ts, xs, '-')
            #ax.axvline(x=new_tstart, color=ls[0].get_color())
        mplutil.save_axes(fig, extra)



