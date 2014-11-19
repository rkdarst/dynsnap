# Richard Darst, May 2012

import os
import re

import matplotlib.figure
import matplotlib.backends.backend_agg
#from matplotlib.patches import CirclePolygon as Circle
import matplotlib.cm as cm
import matplotlib.colors as colors


def get_fig():
    fig = matplotlib.figure.Figure()
    #ax = fig.add_subplot(111, aspect='equal')
    return fig
def write_fig(fig):
    canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    #ax.autoscale_view(tight=True)
    canvas.print_figure(fname, dpi=fig.get_dpi()*dpiScale*escale,
                        bbox_inches='tight')


def get_axes(fname, figsize=(13, 10), ax_hook=None,
             ret_fig=False, figargs={},
             dpi=100):
    """Interface to matplotlib plotting.

    fname: str:
        fname is the string to save to.  It can also have an extension
        of something like FILE.[pdf,png,ps] and it will save a file of
        ALL of these extensions.
    """
    if isinstance(fname, str):
        import matplotlib.figure
        import matplotlib.backends.backend_agg
        fig = matplotlib.figure.Figure(figsize=figsize, dpi=dpi, **figargs)
        canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
        if not ret_fig:
            ax = fig.add_subplot(111)#, aspect='equal')
        else:
            ax = fig
        return ax, (fname, canvas, fig, ax, ax_hook)
    if isinstance(fname, matplotlib.axes.Axes):
        ax = fname
        return ax, (fname, ax_hook)
    else:
        raise ValueError("Unknown type of object: %s"%type(fname))
def save_axes(ax, extra):
    fname = extra[0]
    if isinstance(fname, str):
        fname, canvas, fig, ax, ax_hook = extra
        dirname = os.path.dirname(fname)
        if dirname and not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        multi_ext_match = re.match(r'(.*\.)\[([A-Za-z,]+?)\]$', fname)
        if ax_hook: ax_hook(ax, locals())
        if multi_ext_match:
            # Support for multi-extension matching.  If fname matches
            # FILE.[ext1,ext2], then write the figure to ALL of these
            # files.
            base = multi_ext_match.group(1)
            for ext in multi_ext_match.group(2).split(','):
                canvas.print_figure(base+ext, dpi=fig.get_dpi(),
                                    bbox_inches='tight')
        else:
            canvas.print_figure(fname, dpi=fig.get_dpi(),
                                bbox_inches='tight')
    elif isinstance(fname, matplotlib.axes.Axes):
        fname, ax_hook = extra
        if ax_hook: ax_hook(ax, locals())
    else:
        raise ValueError("Unknown type of object: %s (also, how did we even get to this point?)"%type(fname))


def get_line_style(default='o-',
                   label=None, label_lookup=None,
                   i=None, i_max=None, i_markers=None, colormap=None):
    """
    colormap: str, default None
        colormap name.  Try gist_rainbow.  Requires i and i_max
    """
    # ls, marker, color
    kwargs = { }
    style = default
    # Do an initial update to get starting values of kwargs for use in
    # future configuration.
    if label and label in label_lookup:
        kwargs.update(label_lookup[label])
    else:
        print label
    #print label, label_lookup

    # Custom config of this function through kwargs
    config_vars = ['i', 'i_max', 'colormap']
    if 'i' in kwargs:        i        = kwargs.pop('i')
    if 'i_max' in kwargs:    i_max    = kwargs.pop('i_max')
    if 'colormap' in kwargs: colormap = kwargs.pop('colormap')
    #print colormap, i, i_max
    if i is not None:
        if colormap:
            import pylab as plt
            import numpy
            cm = getattr(plt.cm, colormap)
            colors = cm(numpy.linspace(0, 1, i_max))
            color = colors[i%i_max]
            kwargs['color'] = color
        if i_markers is None:
            kwargs['marker'] = 'o+x<>*^vsh.'[i%11]
        else:
            kwargs['marker'] = i_markers[i%len(i_markers)]
    # This overrides anything else we may set.
    if label and label in label_lookup:
        kwargs.update(label_lookup[label])
    # Remove function-local configuration again:
    for var in config_vars:
        if var in kwargs: kwargs.pop(var)
    return kwargs

class Figure(object):
    def __init__(self, fname, figsize=None):
        self.fname = fname
        if fname:
            import matplotlib.figure
            import matplotlib.backends.backend_agg
            self.fig = matplotlib.figure.Figure(figsize=figsize)
            self.canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(self.fig)
            self.ax = self.fig.add_subplot(111)#, aspect='equal')
        else:
            raise
    def save(self):
        self.canvas.print_figure(self.fname, dpi=self.fig.get_dpi(),
                                 bbox_inches='tight')

