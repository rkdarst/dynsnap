# Richard Darst, November 2015

from __future__ import print_function, division

import numpy as np

import dynsnap
import events
import models

class Viz(object):
    #fname =
    #def model():
    dsargs = [ ]
    figsize = np.asarray([2,2])
    ps = 2
    def __init__(self):
        print("==", self.__class__.__name__)
        evs_list = list(self.model())
        evs = events.Events()
        evs.add_events((t,e,1) for t,e in evs_list)

        _, results = dynsnap.main([None,
                                   #'--dtmode=linear',
                                   #'--dtmax=10',
                                   #'--dtextra=5',
                                   #'--dont-merge-first',
                                   #'--peakfinder=greedy',
                                  ]+self.dsargs,
                                  evs=evs)
        results = results['results']
        if results.thighs[-1] > evs.t_max():
            results.thighs[-1] = evs.t_max()+1

        import pcd.support.matplotlibutil as mplutil

        ax, extra = mplutil.get_axes(self.fname,
                                     figsize=self.figsize)
        ts, es = zip(*evs_list)
        ax.scatter(ts, es, s=self.ps, facecolor='b', edgecolor='b')
        #for thigh in results.thighs[:-1]:
        #    ax.axvline(x=thigh-0.5)
        results.plot_intervals_patches(ax, shift=-.5)

        from matplotlib.font_manager import FontProperties
        font = FontProperties()
        font.set_family('serif')
        ax.set_xlabel('time', fontproperties=font)
        ax.set_ylabel('event ID', fontproperties=font)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale(tight=True)
        if hasattr(self, 'xlim'):  ax.set_xlim(*self.xlim)
        if hasattr(self, 'ylim'):  ax.set_ylim(*self.ylim)

        #axB = ax.twinx()
        #results.plot_Jfinding(axB)

        mplutil.save_axes(ax, extra)
        #import IPython ; IPython.embed()


class Demo00(Viz):
    fname = 'demo00.[pdf,png]'
    ps = 50
    xlim = (-2, 10)
    ylim = (None, 14)
    @staticmethod
    def model():
        return models.demo01(seed=18, p=.6, n=3,
                             t=4, T=8,
                             phase_active=[(0,2),(1,3)])

class Demo01(Viz):
    fname = 'demo01.[pdf,png]'
    @staticmethod
    def model():
        return models.demo01(seed=13)
class Demo01b(Viz):
    fname = 'demo01b.[pdf,png]'
    @staticmethod
    def model():
        return models.demo01(seed=13, n=20, p=.5,
                             T=40,
                             phase_active=[(0,1), (2, 3), (1, 2), (1, 2)])

class Demo02(Viz):
    fname = 'demo02.[pdf,png]'
    dsargs = ['--dont-merge-first']
    @staticmethod
    def model():
        return models.demo02(N=10, seed=14)

class Demo03(Viz):
    fname = 'demo03.[pdf,png]'
    @staticmethod
    def model():
        return models.demo01(seed=14, phase_ps=[0.3, 0.8, 0.3],
                             phase_active=[(0,1), (1,2), (1,2)])

if __name__ == "__main__":
    import sys
    globals()[sys.argv[1]]()
