# Richard Darst, November 2014

import sqlite3

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




import networkx as nx
import pcd.cmtycmp
import pcd.support.algorithms as algs
import numpy

class SnapshotFinder(object):
    old_es = None
    old_g = None
    old_incremental_es = None
    dt_min = 1
    dt_max = 1000
    dt_extra = 50
    def __init__(self, evs):
        self.evs = evs
    def get_first(self, dt):
        es1 = self.evs[self.tstart: self.tstart+dt]
        es1 = set(e for t, e, w in es1)
        es2 = self.evs[self.tstart+dt: self.tstart+2*dt]
        es2 = set(e for t, e, w in es2)
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
                es = self.evs[self.tstart+old_dt : self.tstart+dt]
                old_es.update(set(e for t,e,w in es))
                es = old_es
        if es is None:
            # Recreate edge set from scratch
            es = self.evs[self.tstart: self.tstart+dt]
            es = set(e for t,e,w in es)

        # Cache our results for the next interval
        self.old_incremental_es = (self.tstart, dt, es)

        return es
    def get(self, dt):
        if self.old_es == None:
            return self.get_first(dt)
        else:
            return self.old_es, self.get_succesive(dt)

    def measure_esjacc(self, es1s, es2s):
        den = float(len(es1s | es2s))
        if den == 0:
            x = float('nan')
        else:
            x = len(es1s & es2s) / den
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
        all_dts = range(self.dt_min, self.dt_max+1, 1)
        xs = [ ]
        ts = [ ]
        dts = [ ]
        es1max = es2max = None
        i_max = None
        for i, dt in enumerate(all_dts):
            es1s, es2s = self.get(dt)
            if len(es1s) == 0 or len(es2s) == 0:
                continue
            #es1s = set(es1)
            #es2s = set(es2)
            x = self.measure(es1s, es2s)

            dts.append(dt)
            ts.append(self.tstart+dt)
            xs.append(x)

            i_max = numpy.argmax(xs)
            #if i_max == len(dts)-1:
            #    es1max = set(es1s)
            #    es2max = set(es2s)
            if dt > dts[i_max]+self.dt_extra:
                break
        if i_max is None:
            return None

        self.tried_dts = dts
        self.tried_ts = ts
        self.tried_xs = xs

        #print xs
        dt_max = dts[i_max]
        x_max = xs[i_max]
        es1s, es2s = self.get(dt_max)  # to save correct es2s

        #print "  %4d %3d %3d %s"%(self.tstart, dt_max, i_max, len(dts))
        self.tstart = self.tstart + dt_max
        self.old_es = es2s
        return self.tstart-dt_max, self.tstart

import os
def load_events(fname, col_time=0, col_weight=None, regen=False):
    events = { }
    def _iter():
        f = open(fname)
        for line in f:
            #line = line.split('#', 1)[0]
            line = line.strip()
            if line.startswith('#'): continue
            line = line.split()
            t = line.pop(col_time)
            if col_weight is not None:
                w = line.pop(col_weight)
            else:
                w = 1.0
            e = ' '.join(line)
            if e in events:
                i = events[e]
            else:
                i = len(events)
                events[e] = i
            yield t, i, w
    cache_fname = fname + '.cache'
    if regen:
        # remove existing cache if it exists.
        if os.path.exists(cache_fname):
            os.unlink(cache_fname)
    if os.path.exists(cache_fname):
        ev = Events(cache_fname)
        return ev
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
    parser.add_argument("--regen", action='store_true',
                        help="Recreate temporal event cache")

    parser.add_argument("-t",  type=int, default=0,
                        help="Time column")
    parser.add_argument("-w", type=int, default=None,
                        help="Weight column")
    parser.add_argument("-p", "--plot", action='store_true',
                        help="Plot also?")

    args = parser.parse_args()
    print args

    evs = load_events(args.input, col_time=args.t,
                      col_weight=args.w, regen=args.regen)
    print "file loaded"

    finder = SnapshotFinder(evs)
    finder.tstart = 1000
    #finder.dt_extra = 100
    points = [ ]
    finding_data = [ ]
    if args.output:
        fout = open(args.output+'.txt', 'w')
    while True:
        x = finder.find()
        print finder.tstart
        if x is None:
            break
        finding_data.append((finder.tried_ts, finder.tried_xs, finder.tstart))
        points.append((x[0], x[1]-x[0]))
        points.append((x[1], x[1]-x[0]))
        # Write and record informtion
        if args.output:
            print >> fout, '# t1=%s t2=%s dt=%s'%(x[0], x[1], x[1]-x[0])
            for dt, t, x in zip(finder.tried_dts,
                                finder.tried_ts,
                                finder.tried_xs):
                print >> fout, t, x, dt
            print >> fout
            fout.flush()



    if args.plot:
        import pcd.support.matplotlibutil as mplutil
        fname = args.output+'.[pdf,png]'
        fig, extra = mplutil.get_axes(fname, figsize=(10, 10),
                                      ret_fig=True)
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax.set_xlabel('time')
        #ax2.set_xlim(0, points[-1][0])

        x, y = zip(*points)
        ls = ax.plot(x, y, '-o')
        for ts, xs, new_tstart in finding_data:
            ls = ax2.plot(ts, xs, '-')
            #ax.axvline(x=new_tstart, color=ls[0].get_color())
        mplutil.save_axes(fig, extra)



