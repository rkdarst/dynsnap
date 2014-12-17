from nose.tools import *

import os
import sys

import dynsnap
import models


def run(name, model, output, modelargs={}, kwargs={}):

    events = dynsnap.Events()
    events.add_events((t, e, 1) for t,e in model(modelargs))

    finder = dynsnap.SnapshotFinder(events)

    if 'plot' in kwargs:
        plotter = dynsnap.Plotter(finder)
    while True:
        interval = finder.find()
        if interval is None: break
        t1, t2 = interval

        if 'plot' in kwargs:
            plotter.add(finder)

        print t1, t2

    if 'plot' in kwargs:
        def cb(lcls):
            lcls['ax'].set_title(name)
        plotter.plot(output, callback=cb)


if __name__ == '__main__':

    out_path = 'out-tests/'
    to_run = sys.argv[1:]
    kwargs = dict(plot=True)

    tests = [
        ('toy101A', models.toy101),
        ('toy102A', models.toy102),
        ('toy103A', models.toy103),
        ]

    for test in tests:
        name, model = test
        if to_run and name not in to_run:
            continue

        output = out_path+'test-'+name
        dirname = os.path.dirname(out_path)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        print name, model.func_name
        run(name, model, output, kwargs=kwargs)
