import fnmatch
from functools import partial
import os
import re
import sys

from nose.tools import *

import dynsnap
import models


class BaseTest(object):
    #m = models.model_func
    #ma = {'arg': 'value' }
    desc = None  # description for under the title
    ann_pk = False   # Annotate the peaks in the output plot?
    ma = { }


    def run(self, output, plot=True):
        """Do a run and plot results."""
        # sinec self.m is turned into a bound method.  I don't want to
        # decorate every single subclass with staticmethod... is there
        # a better way to handle this?
        model = self.m.im_func

        events = dynsnap.Events(mode='rw')
        events.add_events((t, e, 1) for t,e in model(**self.ma))

        # Create keyword arguments for the SnapshotFinder from model args.
        varnames = dynsnap.SnapshotFinder.__init__.im_func.func_code.co_varnames
        finder_kwargs = { }
        for name in varnames:
            if name in self.ma:
                finder_kwargs[name] = self.ma[name]

        finder = dynsnap.SnapshotFinder(events,
                                        weighted=self.ma.get('w'),
                                        args=self.ma,
                                        **finder_kwargs)

        if plot:
            results = dynsnap.Results(finder,
                      args=dict(annotate_peaks=self.ann_pk))
        # This is the core.  Iterate through until we are done wwith
        # all intervals.
        x = False
        while True:
            interval = finder.find()
            if interval is None: break
            t1, t2 = interval

            #if not x:
                #x = True
                #for i, t in enumerate(finder._finder_data['ts']):
                #    print finder._finder_data['dts'][i], \
                #          finder._finder_data['measure_data'][i]


            if plot:
                results.add(finder)

            print t1, t2

            #break

        if plot:
            # Callback function to decorate the plot a little bit.
            def cb(lcls):
                title = self.__class__.__name__
                if self.ma:
                    title = "%s (%s)"%(title, " ".join("%s=%s"%(k,v)
                                for k,v in sorted(self.ma.iteritems())))
                if self.desc:
                    title += '\n'+self.desc
                lcls['fig'].suptitle(title)

                # plot theoretical value
                if hasattr(self, 'theory'):
                  for i in (0, 1, 2):
                    if i >= len(lcls['self'].tlows): continue
                    func = self.theory()
                    dat = lcls['self'].finding_data[i]
                    ts, xs = dat
                    ts = ts
                    tlow = lcls['self'].tlows[i]
                    # First round treated differently from future rounds
                    if i == 0:
                        # First round
                        print 'first round'
                        tlow = lcls['self'].tlows[i]
                        dt_prev = 'first'
                    else:
                        dt_prev = lcls['self'].thighs[i-1] - lcls['self'].tlows[i-1]

                    predicted_xs = [ func(dt_prev, t-tlow) for t in ts ]
                    lcls['ax2'].plot(ts, predicted_xs, 'o', color='red')

            results.plot_1(output, callback=cb)


T = BaseTest

class toy101A(T): m=models.toy101
class toy102A(T): m=models.toy102
class toy103A(T):
    m=models.toy103; ma={'seed':13}
class toy103B(T):
    m=models.toy101; ma={'seed':18}
    desc="has some size-8 intervals"

class toy103N(T):
    m=models.toy101; ma={'seed':15}
    desc="upper bound too high"

class drift1Am(T):
    m=models.drift; ma=dict(seed=13, merge_first=True)
class drift1A(T):
    m=models.drift; ma=dict(seed=13, merge_first=False)
class drift1B(T):
    m=models.drift; ma=dict(seed=13, c=0.02, merge_first=False)
class drift1C(T):
    m=models.drift; ma=dict(seed=13, c=0.00, p=.2, merge_first=False,
                            t_max=100, N=10000)
    def theory(self):
        return lambda dt: models.J1(dt, Pe=partial(models.Pe, self.ma['p']))
        #return partial(models.J1_c, self.ma['c'], self.ma['p'])
class drift1D(T):
    m=models.drift; ma=dict(seed=13, t_crit=(200, 500))
class drift1E(T):
    m=models.drift; ma=dict(seed=None, c=0.01, p=.2, merge_first=False,
                            t_max=100, N=100000)
    def theory(self):
        return partial(models.J1_c, self.ma['c'], self.ma['p'])

class drift1F(T):
    from pcd.support.powerlaw import PowerLaw
    cpl = PowerLaw(-.5, xmin=.05, xmax=.8)
    ppl = PowerLaw(-1, xmin=.5, xmax=.8)
    m=models.drift; ma=dict(seed=None, merge_first=False,
                            t_max=1000, N=1000, c_func=cpl.rv, p_func=ppl.rv)

class drift1G(T):
    # Test t_stop at a fixed point.
    m=models.drift; ma=dict(seed=13, c=0.02, merge_first=False,
                            tstop=500)

class drift2A(T):
    # Test t_stop at a fixed point.
    m=models.drift; ma=dict(seed=13, c=0.02, merge_first=False,
                            tstop=500, dtmode='event',
                            )


class periodic1A(T):
    m=models.periodic; ma={'N':1000, 'seed':13}; ann_pk=False
    desc='periodic'
class periodic1Aw(T):
    m=models.periodic; ma={'N':1000, 'seed':13, 'w':True}; ann_pk=False
    desc='periodic - weighted'
class periodic1B(T):
    m=models.periodic; ma={'N':1000, 'seed':13, 't_crit':(200, 500)}
    ann_pk=False
    desc='periodic'



all_tests = sorted((x for name, x in globals().items()
                    if isinstance(x, type)
                        and issubclass(x, BaseTest)
                        and x != BaseTest
                        and not name.startswith('_')),
                    key=lambda x: x.__name__ )

if __name__ == '__main__':

    out_path = 'out-tests/'
    to_run = sys.argv[1:]
    kwargs = dict(plot=True)

    for test in all_tests:
        name = test.__name__

        # Skip tests we don't want to run, if we specify this thing.
        if to_run and not any(re.search(x, name) for x in to_run):
            continue

        output = out_path+'test-'+name
        dirname = os.path.dirname(out_path)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        obj = test()
        print name, obj.m.func_name
        obj.run(output=output, **kwargs)
