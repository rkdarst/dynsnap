"""Basic event handling.

This module provides an Events class, that does the low-level event
storage.  The function ``load_events`` handles most of the logic
related to parsing and caching events.  It has some other analysis
related to events, but this is not used anywhere (including our
papers).

If run on the command line, it provides an interface to the
``load_events`` function.  This is implemented in the ``main``
function (as opposed to the other main_* functions that do other
calculations).
"""

from __future__ import print_function, division

import os
import random
import sqlite3
import sys

# Python 2/3 compatibility.  Avoid dependency on six for one function.
if sys.version_info.major < 3:
    def iteritems(x): return x.iteritems()
else:
    def iteritems(x): return iter(x.items())



class ToInts(object):
    """Convert hashable objects to sequential ints."""
    def __init__(self):
        self._dat = { }
    def __call__(self, x):
        return self._dat.setdefault(x, len(self._dat))
    def __len__(self):
        return len(self._dat)

class Events(object):
    """Event storage database.
    """
    table = 'event'
    def __init__(self, fname=':memory:', mode='r', table=None):
        #if fname == ':memory:' and mode == 'r':
        #    raise ValueError("For fname=':memory:', you probably want mode='rw'")
        if table is not None:
            self.table = table
        elif ':' in fname and os.path.exists(fname.rsplit(':',1)[0]):
            fname, self.table = fname.rsplit(':',1)

        if mode == 'r' and not os.path.exists(fname) and not fname==':memory:':
            raise ValueError("File does not exist: %s"%fname)
        self.conn = sqlite3.connect(fname)

        c = self.conn.cursor()

        c.execute('''CREATE TABLE if not exists %s
                     (t int not null, e int not null, w real)'''%self.table)
        c.execute('''CREATE INDEX if not exists index_%s_t ON %s (t)'''%(
            self.table, self.table))
        #c.execute('''CREATE INDEX if not exists index_event_e ON event (e)''')
        c.execute('''CREATE TABLE if not exists event_name
                     (e INTEGER PRIMARY KEY, name TEXT)''')
        c.execute('''CREATE VIEW IF NOT EXISTS view_%s AS SELECT e, t, w, name
                     FROM %s LEFT JOIN event_name USING (e);'''%(self.table, self.table))
        c.execute('PRAGMA cache_size = -1000000')
        self.conn.commit()
        c.close()
    @classmethod
    def from_iter(cls, it):
        """Create new Events from iter of (t,e,w)"""
        evs = cls()
        evs.add_events(it)
        return evs
    def copy(self):
        """Make a copy of this Events object."""
        evs = Events()
        cur = evs.conn.cursor()
        #for line in self.conn.iterdump():
        #    if line.startswith('CREATE') or line.startswith('COMMIT'): continue
        #    cur.execute(line)
        cur.executemany('INSERT INTO event VALUES (?, ?, ?)', self.conn.execute('SELECT * FROM event'))
        cur.executemany('INSERT INTO event_name VALUES (?, ?)', self.conn.execute('SELECT * FROM event_name'))
        evs.conn.commit()
        #evs.conn.executemany(self.conn.iterdump(), )
        return evs
    def stats(self, convert_t=lambda x: x, tstart=None, tstop=None):
        """Print some event stats"""
        if tstart:
            where = 'where %s <= t AND t < %s'%(tstart, tstop)
        else:
            where = ''
        stats = [ ]
        stats.append(("Number of events",
                      self._execute('select count(w) from %s %s'%(self.table, where)).fetchone()[0]))
        stats.append(("Number of distinct events",
                      self._execute('select count(distinct e) from %s %s'%(self.table, where)).fetchone()[0]))
        stats.append(("First event time",
                      convert_t(self._execute('select min(t) from %s %s'%(self.table, where)).fetchone()[0])))
        stats.append(("Last event time",
                      convert_t(self._execute('select max(t) from %s %s'%(self.table, where)).fetchone()[0])))
        stats.append(("Total count/weight of events",
                      self._execute('select sum(w) from %s %s'%(self.table, where)).fetchone()[0]))
        return stats
    def load(self):
        """Load all events into memory"""
        conn2 = sqlite3.connect(':memory:')
        conn2.executescript("\n".join(self.conn.iterdump()))
        self.conn.close()
        self.conn = conn2

    def _execute(self, stmt, *args):
        c = self.conn.cursor()
        c.execute(stmt, *args)
        return c
    def add_event(self, t, e, w=1.0):
        """Add one event e at time t with weight w."""
        c = self.conn.cursor()
        c.execute("INSERT INTO %s VALUES (?, ?, ?)"%self.table, (t, e, w))
        self.conn.commit()
    def add_events(self, it):
        """Add events from (t,e,w) iterator."""
        c = self.conn.cursor()
        c.executemany("INSERT INTO %s VALUES (?, ?, ?)"%self.table,
                          (x if len(x)==3 else (x[0],x[1],1) for x in it))
        self.conn.commit()
    def t_min(self):
        """min(t) across all data"""
        c = self.conn.cursor()
        c.execute("SELECT min(t) from %s"%self.table)
        return c.fetchone()[0]
    def t_max(self):
        """max(t) across all data"""
        c = self.conn.cursor()
        c.execute("SELECT max(t) from %s"%self.table)
        return c.fetchone()[0]
    def times_next(self, tmin, range=None):
        """Return iterator of distinct event times t>tmin."""
        c = self.conn.cursor()
        if range:
            c.execute("SELECT distinct t from %s where ?<t<=? "
                      "order by t"%self.table, (tmin, tmin+range))
        else:
            c.execute("SELECT distinct t from %s where ?<t "
                      "order by t"%self.table, (tmin, ))
        for row in c:
            yield row[0]
    def dump(self):
        """Dump all data.  Prints (t,e,w) rows."""
        c = self.conn.cursor()
        c.execute("SELECT t,e,w from %s"%self.table)
        for row in c:
            print(row[0], row[1], row[2])

    # Basic properties.
    def __len__(self):
        """Total number of data points"""
        c = self.conn.cursor()
        c.execute("SELECT count(*) from %s"%self.table)
        return c.fetchone()[0]
    def n_distinct_events(self):
        """Number of distinct event IDs in data"""
        c = self.conn.cursor()
        c.execute("SELECT count( DISTINCT e ) from %s"%self.table)
        return c.fetchone()[0]
    def count_interval(self, low, high):
        """Number of points low <= t < thigh"""
        c = self.conn.cursor()
        c.execute("SELECT count(*) FROM %s WHERE ?<=t AND t <?"%self.table, (low, high))
        return c.fetchone()[0]
    # Iterators of events
    def iter_distinct_events(self):
        """Iterate over all distinct integer events IDs e"""
        c = self.conn.cursor()
        c.execute("SELECT DISTINCT e from %s"%self.table)
        for (e,) in c:
            yield e
    def iter_ordered_of_event(self, e):
        """Iterate over all (t,w) instances of event e."""
        c = self.conn.cursor()
        c.execute("SELECT DISTINCT t, w from %s where e=?"%self.table, (e, ))
        return c
    def __getitem__(self, interval):
        """Return a slice: iterator of (t,e,w) in interval.

        Usage:
            it events[tlow:thigh]

            _tmp = events[tlow:]
            it = _tmp[:thigh]

        Returns iterator of (t,e,w) in events.
        """
        assert isinstance(interval, slice)
        assert interval.step is None
        assert interval.start is not None, "Must specify interval start"
        c = self.conn.cursor()
        if interval.stop is not None:
            # Actual slice.
            c.execute('''select t, e, w from %s where ? <= t AND t < ?'''%self.table,
                      (interval.start, interval.stop, ))
            return c
        else:
            return _EventListSubset(self, interval.start)
    def iter_ordered(self, weights=True):
        """Iter all (t,e,w) or (t,e) tuples."""
        c = self.conn.cursor()
        if weights:
            c.execute('''select t, e, w from %s order by t'''%self.table)
        else:
            c.execute('''select t, e from %s order by t'''%self.table)
        return c
    def event_density(self, domain, halfwidth, low=None, high=None):
        """Return a local average of event density"""
        #tlow = math.floor(tlow/interval)*interval
        #thigh = math.ceil(thigh/interval)*interval
        #domain = numpy.range(tlow, thigh+interval, interval)
        c = self.conn.cursor()
        if low  is None: low  = halfwidth
        if high is None: high = halfwidth
        vals = [ c.execute('''select sum(w) from %s where ? <= t AND t < ?'''%self.table,
                      (x-low, x+high, )).fetchone()[0]
                 for x in domain]
        return domain, vals

    def add_event_names(self, it):
        """Store event names in database.

        it: iterator of (event_name, event_id) tuples."""
        if isinstance(it, dict):
            it = ((eid, repr(ename)) for ename, eid in iteritems(it))
        c = self.conn.cursor()
        c.executemany('''INSERT INTO event_name VALUES (?, ?)''', it)
        self.conn.commit()
    def iter_event_names(self):
        c = self.conn.cursor()
        c.execute("""SELECT e, name FROM event_name""")
        return c
    def get_event_names(self, it):
        c = self.conn.cursor()
        if hasattr(it, '__iter__'):
            names = [ ]
            for eid in it:
                row = c.execute("""SELECT name FROM event_name WHERE e=?""", (eid,)).fetchone()
                if not row: continue
                names.append(row[0])
            return names
        names = c.execute("""SELECT name FROM event_name WHERE e=?""",
                         (it, )).fetchall()
        return names[0][0]
    # Randomization
    def shuffle_e(self):
        """Shuffle all event IDs, preserving timestamps and weights."""
        c = self.conn.cursor()
        rowids, es = zip(*self._execute('SELECT rowid, e FROM event'))
        es = list(es)
        random.shuffle(es)
        c.executemany('UPDATE event SET e=? WHERE rowid=?', zip(es, rowids))
        self.conn.commit()
    randomize_order = shuffle_e
    def shuffle_w(self):
        """Shuffle all weights, preserving event IDs and timestamps."""
        c = self.conn.cursor()
        rowids, ws = zip(*c.execute('SELECT rowid, w FROM event'))
        ws = list(ws)
        random.shuffle(ws)
        c.executemany('UPDATE event SET w=? WHERE rowid=?', zip(ws, rowids))
        self.conn.commit()
    def shuffle_ew(self):
        """Shuffle all event IDs and weights together.

        shuffles all (event, weight) pairs, but keeps original
        timestamps.  The effect is the same as randomly interchanging
        all timestamps."""
        c = self.conn.cursor()
        rowids, es, ws = zip(*c.execute('SELECT rowid, e, w FROM event'))
        items = list(zip(es, ws))
        random.shuffle(items)
        es, ws = zip(*items)
        c.executemany('UPDATE event SET e=?, w=? WHERE rowid=?', zip(es, ws, rowids))
        self.conn.commit()
    def shuffle_t1(self):
        """Shuffle timestamps while preserving order.

        All event IDs, weights are the same and in the same order, but
        timestamps are randomly created from a uniform distribution.
        """
        t_min = self.t_min()
        t_max = self.t_max()
        c = self.conn.cursor()
        rowids, = zip(*c.execute('SELECT rowid FROM event ORDER BY t'))
        ts = [ random.uniform(t_min, t_max) for _ in range(len(rowids)) ]
        ts.sort()
        c.executemany('UPDATE event SET t=? WHERE rowid=?', zip(ts, rowids))
        self.conn.commit()
    def randomize_eid(self, new_es=None):
        """Randomize event IDs: keep all events the same"""
        es, = zip(*self._execute('SELECT DISTINCT e FROM event order by e'))
        eid_min = min(es)
        eid_max = max(es)
        if new_es is None:
            new_es = list(range(eid_max+1, eid_max+1+len(es)))
            random.shuffle(new_es)
        elif new_es == 'order':
            es = list(x[0] for x in self._execute('SELECT e FROM event GROUP BY e ORDER BY min(t)'))
            new_es = list(range(eid_max+1, eid_max+1+len(es)))
        #print("mapping=", list(zip(es, new_es)))
        self.conn.executemany('UPDATE %s SET e=? WHERE e=?'%self.table,
                         zip(new_es, es))
        self.conn.executemany('UPDATE event_name SET e=? WHERE e=?',
                         zip(new_es, es))
        self.conn.commit()


class _EventListSubset(object):
    """This is used for partial evaluation of Events.__getitem__"""
    def __init__(self, events, t_start):
        self.events = events
        self.t_start = t_start
    def __getitem__(self, interval):
        assert isinstance(interval, slice)
        assert interval.start is None
        assert interval.stop is not None

        c = self.events.conn.execute(
            '''select t, e, w from %s where ? <= t AND t < ?'''%self.events.table,
            (self.t_start, interval.stop, ))
        return c


import ast
import os
def load_events(fname, col_time=0, col_weight=None, cache=False, regen=False,
                unordered=False, grouped=False, cache_fname=None,
                cols_data=None, separator=None):
    """Load events from file into an Events object.

    This function serves multiple purposes.  With just an input file,
    it will parse and load the events.  It can also handle caching
    those events to an sqlite3 database, opening those caches, and has
    logic to automatically finding and loading caches if they exist.
    In the future, some of these functions may separate from each
    other, because this function does a little bit too much.

    """
    # Shortcut: if file already exists and is an Events object, simply
    # return that and don't do anything.  This is what implements the
    # from-cache loader system.
    try:
        evs = Events(fname, mode='r')
        return evs
    except sqlite3.DatabaseError:
        pass

    # If cols_data is a string, convert to a tuple like is needed.
    if cols_data and isinstance(cols_data, str):
        cols_data = tuple(int(x) for x in cols_data.split(','))

    # Primary loop that reads the input file.  This operates as an
    # iterator, and no code is executed directly in the next block.
    # It also alters the 'events' dictionary, which converts things to
    # ints.
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
            line = line.split(separator)
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

    # This logic handles opening and/or regenerating an existing
    # cache.  The filename is not usually specified.  This should be
    # eventually moved outside of this function.
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
    # This is where all the actual parsing happens.  Run through the
    # iterator, loading it into sqlite3, and then save the original event names.
    ev = Events(cache_fname, mode='rw')
    ev.add_events(_iter())
    ev.add_event_names(events)
    return ev

def main_cache(argv=sys.argv):
    """Main function for event caching"""
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
                        help="Columns containing the data"
                            " (comma separated list)")
    parser.add_argument("--separator", default=None,
                        help="Column separator (default any whitespace)")

    args = parser.parse_args(argv[1:])


    evs = load_events(args.input, col_time=args.t,
                      col_weight=args.w, cache_fname=args.cached_output,
                      unordered=args.unordered,
                      grouped=args.grouped,
                      cols_data=args.datacols,
                      cache=True,
                      separator=args.separator)


def main_summary(argv=sys.argv):
    """Main function for producing other event summaries"""
    evs = Events(argv[1])
    print("Number of events: ", len(evs))
    print("Number of unique events: ", evs.n_distinct_events())
    print("t_min, t_max:", evs.t_min(), evs.t_max())


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
    print(len(event_ts))
    print(len(evs))
    print(np.mean(burstinesses), np.std(burstinesses))

def _burstiness_do(elist):
    print('pre-start')
    evs = Events(fname)
    #print elist
    print('start')
    lists = [ list(t for t,w in evs.iter_ordered_of_event(e)) for e in elist ]
    x =  [ burstiness(l) for l in lists if len(l) > 1 ]
    print('stop')
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
    print(len(events_grouped))

    burstinesses = [ ]

    from multiprocessing import Pool
    pool = Pool(processes=1)
    #for elist in events_grouped:
    #    result = pool.apply_async(_burstiness_do, elist,
    #                              callback=lambda x: burstinesses.extend(x))
    results = pool.map(_burstiness_do, events_grouped)
    print(type(results))
    for e in results:
        burstinesses.extend(e)
    pool.close()
    pool.join()
    print(evs.n_distinct_events())
    print(len(burstinesses))
    print(np.mean(burstinesses), np.std(burstinesses))




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
        main_cache()
