from nose.tools import *

import os
import sys

import dynsnap
import models


def run(name, model, output, modelargs={}, kwargs={}):

    events = dynsnap.Events()
    events.add_events((t, e, 1) for t,e in model(**modelargs))

    finder = dynsnap.SnapshotFinder(events,
                                    weighted=modelargs.get('w'))

    if 'plot' in kwargs:
        plotter = dynsnap.Plotter(finder,
                  args=dict(annotate_peaks=kwargs.get('annotate_peaks', True)))
    while True:
        interval = finder.find()
        if interval is None: break
        t1, t2 = interval

        if 'plot' in kwargs:
            plotter.add(finder)

        print t1, t2

    if 'plot' in kwargs:
        def cb(lcls):
            title = name
            if modelargs:
                title = "%s (%s)"%(title, " ".join("%s=%s"%(k,v)
                            for k,v in sorted(modelargs.iteritems())))
            if 'desc' in kwargs:
                title += '\n'+kwargs['desc']
            lcls['fig'].suptitle(title)

        plotter.plot(output, callback=cb)


if __name__ == '__main__':

    out_path = 'out-tests/'
    to_run = sys.argv[1:]
    kwargs_ = dict(plot=True)

    tests = [
        ('toy101A', models.toy101),
        ('toy102A', models.toy102),
        ('toy103A', models.toy103, dict(seed=13)),
        ('toy103B', models.toy103, dict(seed=18,),
                                   dict(desc="has some size-8 intervals")),

        ('toy103N', models.toy103, dict(seed=15),
                                   dict(desc="upper bound too high")),


        ('periodic1A', models.periodic, dict(N=1000, seed=13),
                                        dict(desc="periodic",
                                             annotate_peaks=False)),
        ('periodic1Aw', models.periodic, dict(N=1000, seed=13, w=True),
                                         dict(desc="periodic",
                                             annotate_peaks=False)),

        ]

    for testdat in tests:
        name, model = testdat[:2]
        kwargs = kwargs_.copy()
        modelargs = { }
        if len(testdat) >= 3:
            modelargs = testdat[2]
        if len(testdat) >= 4:
            kwargs.update(testdat[3])

        # Skip tests we don't want to run, if we specify this thing.
        if to_run and name not in to_run:
            continue

        output = out_path+'test-'+name
        dirname = os.path.dirname(out_path)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        print name, model.func_name
        run(name, model, output, modelargs=modelargs, kwargs=kwargs)
