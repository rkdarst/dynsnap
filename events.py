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

def main():
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

    args = parser.parse_args()


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


if __name__ == '__main__':
    main()
