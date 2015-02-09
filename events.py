import os
import sqlite3
import sys

class Events(object):
    def __init__(self, fname=':memory:', mode='r'):
        if mode == 'r' and not os.path.exists(fname):
            raise ValueError("File does not exist: %s"%fname)
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
    def times_next(self, tmin, range=None):
        """Return distinct times after t."""
        c = self.conn.cursor()
        if range:
            c.execute("SELECT distinct t from event where ?<t<=? "
                      "order by t", (tmin, tmin+range))
        else:
            c.execute("SELECT distinct t from event where ?<t "
                      "order by t", (tmin, ))
        for row in c:
            yield row[0]
    def dump(self):
        c = self.conn.cursor()
        c.execute("SELECT t,e,w from event")
        for row in c:
            print row[0], row[1], row[2]

    def __len__(self):
        c = self.conn.cursor()
        c.execute("SELECT count(*) from event")
        return c.fetchone()[0]
    def n_distinct_events(self):
        c = self.conn.cursor()
        c.execute("SELECT count( DISTINCT e ) from event")
        return c.fetchone()[0]
    def iter_distinct_events(self):
        c = self.conn.cursor()
        c.execute("SELECT DISTINCT e from event")
        for (e,) in c:
            yield e
    def iter_ordered_of_event(self, e):
        c = self.conn.cursor()
        c.execute("SELECT DISTINCT t, w from event where e=?", (e, ))
        return c


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
    def iter_ordered(self):
        c = self.conn.cursor()
        c.execute('''select t, e, w from event order by t''')
        return c


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


import ast
import os
def load_events(fname, col_time=0, col_weight=None, cache=False, regen=False,
                unordered=False, grouped=False, cache_fname=None,
                cols_data=None):
    try:
        evs = Events(fname)
        return evs
    except sqlite3.DatabaseError:
        pass

    events = { }
    def _iter():
        if fname == '-':
            import sys
            f = sys.stdin
        else:
            f = open(fname)
        for lineno, line in enumerate(f):
            #if lineno > 10000000:
            #    break
            #line = line.split('#', 1)[0]
            line = line.strip()
            if line.startswith('#'): continue
            line = line.split()
            #line_orig = tuple(line)
            if not line: continue
            t = ast.literal_eval(line[col_time])
            #col_weight2 = col_weight # modified in this scope so needs
            #                         # local copy
            if col_weight is not None and col_weight != -1:
                assert col_weight != col_time, ("weight column specified "
                                                "same as time column.")
                #if col_weight > col_time:
                #    # We removed one column, need to adjust.
                #    col_weight2 -= 1
                w = ast.literal_eval(line[col_weight])
            else:
                w = 1.0

            if cols_data:
                line = tuple( line[i] for i in cols_data )
                #print cols_data, line
            else:
                line = tuple( x for i,x in enumerate(line) if i!=col_time and i!=col_weight )

            if grouped:
                # each line contains many different events.  Handle
                # his case, then coninue the loop.
                for e in line:
                    if e in events:
                        i = events[e]
                    else:
                        i = len(events)
                        events[e] = i
                    yield t, i, w
                continue

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
        if cache_fname is None:
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

def main(argv=sys.argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="benchmark model to simulate",)
    parser.add_argument("cached_output", help="Write cache to this file")


    parser.add_argument("--unordered", action='store_true',
                        help="Event columns on the line are unordered")
    parser.add_argument("--grouped", action='store_true',
                        help="Each line contains different space-separated "
                        "events.")
    parser.add_argument("-t",  type=int, default=0,
                        help="Time column")
    parser.add_argument("-w", type=int, default=None,
                        help="Weight column")
    parser.add_argument("--datacols", default="",
                        help="Weight column")

    args = parser.parse_args(argv[1:])


    if args.datacols:
        datacols = tuple(int(x) for x in args.datacols.split(','))
    else:
        datacols = None
    evs = load_events(args.input, col_time=args.t,
                      col_weight=args.w, cache_fname=args.cached_output,
                      unordered=args.unordered,
                      grouped=args.grouped,
                      cols_data=datacols,
                      cache=True)


def main_summary(argv=sys.argv):
    evs = Events(argv[1])
    print "Number of events: ", len(evs)
    print "Number of unique events: ", evs.n_distinct_events()
    print "t_min, t_max:", evs.t_min(), evs.t_max()


def main_analyze(argv=sys.argv):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")


    parser.add_argument("--uw", action='store_true', help="Weighted analysis")
    parser.add_argument("--timescale", type=float, help="time scale")
    parser.add_argument("--nitems", type=int, help="number of items to analyze")
    args = parser.parse_args(argv[1:])

    time_scale = 1.
    if args.timescale:
        time_scale = args.timescale
    weighted = not args.uw

    evs = Events(args.input)
    e_last_time = { }
    inter_event_times = [ ]
    t_min = evs.t_min()
    nitems = args.nitems

    for i, (t, e, w) in enumerate(evs.iter_ordered()):
        if nitems and i > nitems:
            break
        #print t, e, w
        if e not in e_last_time:
            e_last_time[e] = t
            inter_event_times.append(t-t_min)
            continue
        dt = t - e_last_time[e]
        inter_event_times.append(dt)

    for t_lastseen in e_last_time.itervalues():
        inter_event_times.append(t-t_lastseen)


    if weighted:
        weights = inter_event_times
    else:
        weights = None
    import numpy
    hist, bin_edges = numpy.histogram(inter_event_times,
                                      weights=weights,
                                      bins=50, normed=True)
    #print hist
    #print bin_edges

    import pcd.support.matplotlibutil as mplutil

    bin_edges /= time_scale

    ax, extra = mplutil.get_axes(args.output+'.[png,pdf]')
    ax.plot(bin_edges[:-1], hist)
    ax.set_xlabel('$\Delta t$')
    ax.set_ylabel('PDF')
    ax.set_title('total t: [%s, %s], Dt=%4.2f'%(t_min, t, (t-t_min)/time_scale))
    mplutil.save_axes(ax, extra)


import numpy as np
def getInterEventTime(tArray, dataDuration, periodicBoundary=False):
    iet = np.diff(tArray)
    if periodicBoundary is True:
        iet = np.r_[iet, dataDuration+tArray[0]-tArray[-1]]
    return iet
def burstiness(tArray, dataDuration, periodicBoundary=True):
    iet = getInterEventTime(tArray, periodicBoundary, dataDuration)
    sigma_t = np.std(iet)
    mu_t = np.mean(iet)
    return (sigma_t-mu_t)/(sigma_t+mu_t)


def main_burstiness(argv=sys.argv):
    evs = Events(argv[1])

    # The following two functions are from Raj in
    # /networks/rajkp/Research/Network/Temporal.Networks/Codes/temporalProperties.py

    import collections
    event_ts = collections.defaultdict(list)
    for e, t, w in evs.iter_ordered():
        event_ts[e].append(t)

    duration = evs.t_max() - evs.t_min()

    burstinesses = [ burstiness(tArray, dataDuration=duration, periodicBoundary=True)
                     for tArray in event_ts.itervalues()
                     if len(tArray) > 1 ]
    print len(event_ts)
    print len(evs)
    print np.mean(burstinesses), np.std(burstinesses)

def _burstiness_do(elist):
    print 'pre-start'
    evs = Events(fname)
    #print elist
    print 'start'
    lists = [ list(t for t,w in evs.iter_ordered_of_event(e)) for e in elist ]
    x =  [ burstiness(l) for l in lists if len(l) > 1 ]
    print 'stop'
    return x

def main_burstiness_parallel(argv=sys.argv):
    global fname
    fname = argv[1]
    evs = Events(argv[1])

    event_iter = evs.iter_distinct_events()
    events_grouped = [ ]
    try:
      while True:
        elist = [ ]
        events_grouped.append(elist)
        for _ in range(500):
            elist.append(next(event_iter))
    except StopIteration:
        pass
    print len(events_grouped)

    burstinesses = [ ]

    from multiprocessing import Pool
    pool = Pool(processes=1)
    #for elist in events_grouped:
    #    result = pool.apply_async(_burstiness_do, elist,
    #                              callback=lambda x: burstinesses.extend(x))
    results = pool.map(_burstiness_do, events_grouped)
    print type(results)
    for e in results:
        burstinesses.extend(e)
    pool.close()
    pool.join()
    print evs.n_distinct_events()
    print len(burstinesses)
    print np.mean(burstinesses), np.std(burstinesses)




if __name__ == '__main__':
    if sys.argv[1] == 'analyze':
        main_analyze(argv=sys.argv[0:1]+sys.argv[2:])
    elif sys.argv[1] == 'summary':
        main_summary(argv=sys.argv[0:1]+sys.argv[2:])
    elif sys.argv[1] == 'burstiness':
        main_burstiness(argv=sys.argv[0:1]+sys.argv[2:])
    elif 'main_'+sys.argv[1] in globals():
        globals()['main_'+sys.argv[1]](argv=sys.argv[0:1]+sys.argv[2:])
    else:
        main()
