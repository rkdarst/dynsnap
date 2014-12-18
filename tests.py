from nose.tools import *

import os
import sys

import dynsnap
import models


class BaseTest(object):
    #m = models.model_func
    #ma = {'arg': 'value' }
    #desc = 'description for under subplot'
    ann_pk = True   # Annotate the peaks in the output plot?


    def run(self, output, plot=True):
        """Do a run and plot results."""
        # sinec self.m is turned into a bound method.  I don't want to
        # decorate every single subclass with staticmethod... is there
        # a better way to handle this?
        model = self.m.im_func

        events = dynsnap.Events()
        events.add_events((t, e, 1) for t,e in model(**self.ma))

        finder = dynsnap.SnapshotFinder(events,
                                        weighted=self.ma.get('w'))

        if plot:
            plotter = dynsnap.Plotter(finder,
                      args=dict(annotate_peaks=self.ann_pk))
        # This is the core.  Iterate through until we are done wwith
        # all intervals.
        while True:
            interval = finder.find()
            if interval is None: break
            t1, t2 = interval

            if plot:
                plotter.add(finder)

            print t1, t2

        if plot:
            # Callback function to decorate the plot a little bit.
            def cb(lcls):
                title = self.__class__.__name__
                if self.ma:
                    title = "%s (%s)"%(title, " ".join("%s=%s"%(k,v)
                                for k,v in sorted(self.ma.iteritems())))
                if 'desc' in kwargs:
                    title += '\n'+kwargs['desc']
                lcls['fig'].suptitle(title)

            plotter.plot(output, callback=cb)


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

class periodic1A(T):
    m=models.periodic; ma={'N':1000, 'seed':13}; ann_pk=False
    desc='periodic'
class periodic1Aw(T):
    m=models.periodic; ma={'N':1000, 'seed':13, 'w':True}; ann_pk=False
    desc='periodic - weighted'


all_tests = [x for name, x in globals().items()
             if isinstance(x, type) and issubclass(x, BaseTest) and x != BaseTest
             and not name.startswith('_') ]

if __name__ == '__main__':

    out_path = 'out-tests/'
    to_run = sys.argv[1:]
    kwargs = dict(plot=True)

    for test in all_tests:
        name = test.__name__

        # Skip tests we don't want to run, if we specify this thing.
        if to_run and name not in to_run:
            continue


        output = out_path+'test-'+name
        dirname = os.path.dirname(out_path)
        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname)

        obj = test()
        print name, obj.m.func_name
        obj.run(output=output, **kwargs)
