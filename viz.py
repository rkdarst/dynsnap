# Richard Darst, November 2015

import dynsnap
import events
import models

class Viz(object):
    #fname =
    #def model():
    dsargs = [ ]
    def __init__(self):
        evs_list = list(self.model())
        evs = events.Events()
        evs.add_events((t,e,1) for t,e in evs_list)

        _, results = dynsnap.main([None,
                                   '--dtmode=linear',
                                   '--dtmax=10',
                                   '--dtextra=5',
                                   #'--dont-merge-first'
                                  ]+self.dsargs,
                                  evs=evs)
        results = results['results']

        import pcd.support.matplotlibutil as mplutil

        ax, extra = mplutil.get_axes(self.fname)
        ts, es = zip(*evs_list)
        ax.scatter(ts, es, s=80)
        for thigh in results.thighs[:-1]:
            ax.axvline(x=thigh-0.5)

        mplutil.save_axes(ax, extra)
        #import IPython ; IPython.embed()


#class Demo00(Viz):
#    fname = 'demo00.pdf'
#    @staticmethod
#    def model():
#        return models.demo01(seed=13)

class Demo01(Viz):
    fname = 'demo01.pdf'
    @staticmethod
    def model():
        return models.demo01(seed=13)

class Demo02(Viz):
    fname = 'demo02.pdf'
    dsargs = ['--dont-merge-first']
    @staticmethod
    def model():
        return models.demo02(N=10, seed=13)

class Demo03(Viz):
    fname = 'demo03.pdf'
    @staticmethod
    def model():
        return models.demo01(seed=14, phase_ps=[0.3, 0.8, 0.3],
                             phase_active=[(0,1), (1,2), (1,2)])

if __name__ == "__main__":
    import sys
    globals()[sys.argv[1]]()
