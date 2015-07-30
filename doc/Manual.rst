Dynamic segmentation tool manual
================================

This tool dynamically finds segments in data.



Basic principles
----------------

This code takes a list of events (each event occurring at a certain
time), and segments the events into time intervals.

The first input is some ``InputFile``.  The most basic format has one
column for time, and all remaining columns are interpreted as strings,
and specify the event.

For efficiency reasons, the input file can be compiled into a sqlite
database using ``events.py``.  This allows optimized lookups on all
events, and ability to quickly re-run with different options.  This
happens internally when segmenting, anyway, so you may as well do it
yourself first.

Finally, one would use ``dynsnap.py`` to run on the compiled data (or
raw data, if you skipped that step), and produce some text output
files, and optionally plot files.  These would need to be analyzed by
yourself to understand what is going on.




Example usage
-------------

This is not fully explanatory, but provides an idea of how this tool
is used.

Input file::

   0 1 A B
   0 2 A C
   1 1 B D
   1 4 B C
   ...

In this file, the first column is the time, the second column is some
event weight (optional), and other columns are the event identifiers.  In this
case, the event identifiers look something like edges, with two
events.  The events are ordered tuples (A, B), etc.  (There are extra
options for unordered tuples or selecting only certain columns).

Now, we process this into the optimized sqlite database::

    python events.py input.txt input.sqlite -t 0 -w 1

``input.txt`` and ``input.sqlite`` names are input/output filenames.
``-t 0`` indicates that time is in the zeroth column.  ``-w 1``
indicates that the first (zero-indexed) column contains weights.
These weights are recorded within the database, but to specify a
weighed analysis you need to give the weighted option to the next
step, too.  Weights default to 1 if ``-w`` is not specified.

Then, we can run the actual analysis::

    python dynsnap.py input.sqlite input.out -p -w -1

The ``-p`` option indicates that we want to create some plots, in
addition to the standard output files.  You must include the ``-w -1``
option here to get the weighted analysis, but the column number is
irrelevant since weights are already stored in the database (so ``-1``
is used).

Several output files are written, starting with ``input.out``.

input.out.txt:
    Contains columns ``t_low t_high interval_length similarity_value ...``

input.out.J.txt:
    Contains information on the maximization process.

input.out.{pdf, png}:
    Plots of the segmentation.




Command line usage
------------------

dynsnap.py - finding segment intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This program does the actual analysis.  The most efficient method is
to first use ``events.py`` to create a database of events, and run
this program on that database.  However, you can run directly on input
.txt files, too, and for that some options of ``events.py`` are
duplicated here.

There are very many options for controlling this process.  While a
description of options is given here, details are in the accompanying
scientific README.


Basic syntax::

   python dynsnap.py {InputFile} {OutputName} [options]

Arguments:

:InputFile:
    Raw input file, or processed .sqlite input file.

:OutputName:
    Prefix for output files.  Output is written to OutputName.out.txt,
    etc.

User interaction options:

--plot, -p
    Write some plots at ``OutputName.{pdf,png}``.  Requires matplotlib
    and ``pcd``.
--plotstyle
    Format of the plot style.  ``1``, ``2``, or ``3`` for different
    plotting formats.  These are designed for quick information, not
    to be ready for future use.
--interact
    If given, start an interactive IPython shell at the end of the
    main function.  This
--tstart T
    Start analysis only after this time.  Events before this time are
    not segmented, and the first segment starts at this time.
    Argument should be a number matching the numeric time scale.
--tstop T
    Stop analysis after this time.  Events after this time are not
    segmented.  Same format as ``--tstart``.
--tformat=FORMAT
    Options available: ``unixtime``.  The time number is interpreted
    according to this rule, and converted into the date and time.
    This is used for plotting and also for printing to stdout and the
    output files.
--stats
    Don't do anything, just print some stats on the data and exit.
    Perhaps this option would be better placed in ``events.py``, but
    here it can use the values from ``--tstart``, ``--tstop``, and
    ``--tformat`` to make the results more useful.


Options related to the segmentation algorithm:

-w N
    Use as ``-w -1`` if running without a text file input.  Turn on
    weighted analysis, even if not reading from an input text file.
    Normally, this is used to specify the weight column in input
    files.  However, if data is stored in the database but ``-w -1``
    is not given, unweighted sets will be used and the weights are
    lost.  Basically, make sure that some form of the ``-w`` option is
    on the command line when you want to do use a weighted similarity
    measure.
--measure
    Specify similarity measure to use.  Options are ``jacc``,
    ``cosine``, or ``cosine_uw`` (unweighted).  Unfortunately, these
    measures interact with the ``-w`` option.  The following explains
    how to use each different measure.

    unweighted Jaccard
        Default option.  Do *not* specify ``-w``.
    weighted Jaccard
        Must specify ``-w`` with a column value or ``-1``.
    Cosine similarity
        ``--measure=cosine -w -1`` (or a column number for the weight option)
    Cosine similarity, unweighted
        ``--measure=cosine_uw``.
--dont-merge-first
    Do not perform the "merge first two intervals" process.  By
    default, the first two intervals are merged.  It is recommended to
    disable this.
--dtmode=NAME
    Select among the three types of search patterns: ``linear``,
    ``log``, and ``event``.  The default is ``log`` and this has been
    adapted to suit almost any data.

    linear:
        Simple, dumb linear search.  Set ``--dtstep=STEP_SIZE`` to
        adjust scale, and optionally ``--dtmin=``, ``--dtmax=``, and
        ``--dtextra=`` to control other parameters of the search.

    event:
        Scans exactly the dt intervals corresponding to the next
        events.  This adapts to the scale of the data, but is still
        inefficient if the optimal time scale is much larger than
        the inter-event time.

    log:
        Logarithmic scanning.  Scans 1, 2, .. 99, 100, 110, 120, ...,
        990, 1000, 1100, ....  This is scaled by a power of 10 to
	match the size of the first next event.

--peakfinder=NAME
    Method of finding peaks of similarity, if there is a plateau of
    the same value.  Options are ``longest``, ``shortest``,
    ``greedy``.  The default is ``longest``.

    longest:
        longest-time plateau value.
    shortest:
        shortest-time plateau value.

    greedy:
        A greedy search for longest plateau value.  As soon as the
        first decrease is detected, abort the search and use the
        longest plateau value.  This is in contrast to ``longest``
        and ``shortest``, which go a bit further and make sure
        that there is no future greater maximum.


Options for --dtmode=linear

--dtstep
    Set the increment for searching.  Only for the ``linear`` scan
    mode.  Default 1.
--dtmin
    Set the minimum search time.  Only for the ``linear`` scan mode.
    Default 1.
--dtmax
    Set the maximum search time.  Only for the ``linear`` scan mode.
    Default 1.
--dtextra
    After a peak is found, search this much further in time before
    settling on the peak.  By default, an adaptive method is used.
--log-dtmin
    Set the increment for searching.  Only for the ``log`` scan mode.  Default 1.
--log-dtmax
    Not used.

Options related to input.  These options relate to data input, and
have the same usage as in ``events.py``.  See that section for full
information.  Column numbers start from 0.

-w N
    Specify weighted analysis.  If operating on raw input, the
    zero-indexed column number of the weights in the file.  If
    operating on an sqlite database, specify ``-1`` or anything to
    turn on weighted analysis.
-t N
    Time column
--unordered
    Data columns should be treated as unordered.
--grouped
    See documentation for events.py below.
--datacols
    Data columns.
--cache
    If given, ``dynsnap.py`` operates a bit like ``event.py`` so that
    the initial data is stored in an sqlite database, with a name
    based on the input filename.  If the cache already exists and this
    option is given, use that cache and don't re-read the original
    data.  Note that the data-related options thus have no effect
    (except ``-w -1``).  Recommend to compile using ``events.py``
    since there is less risk of unexpected behavior.  This option is
    deprecated and will be removed eventually.
--regen
    Delete and regenerate the cache.  Only has any effect if
    ``--cache`` is specified.


events.py -- preprocess input files into an optimized database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This program will compile an input file into an optimized sqlite
database of events.  This is used to make runs of the segmentation
faster, since in general data doesn't change, but segmentation is
often re-run with different options.

The output is an ``sqlite`` database self-contained within one file.
It can be examined using the ``sqlite3`` command line utility, and the
format is somewhat self-explanatory.

Basic syntax::

   python events.py {InputFile} {OutputName} [options]

Arguments:

:InputFile:
    Raw input file.  Should be a space-separated text file.  See
    the section Input Formats for more information.
:OutputName:
    Output sqlite database.

-t N
    Specify column holding time.  Columns are zero-indexed!
    Default: 0.
-w N
    Specify column holding weight.  Columns are zero-indexed!
    Default: None.
--unordered
    If given, the ordering of other columns does not matter, and lines
    with events "aaa bbb ccc" and "aaa ccc bbb" are considered the
    same.  This, for example, makes graph edges be considered as
    undirected.
--grouped
    Alternative input format where each line has multiple
    space-separated events.  See the section Input Formats.
--datacols
    If given, only these columns are considered for specifying
    events.  All other columns (besides the time and weight columns)
    are ignored.


models.py
~~~~~~~~~

This is an interface to various toy models.  Run the program with a
name of a model to generate output on ``stdout``.  The ``--grouped``
option can be given to produce output in grouped format (see below).

Syntax::

  python models.py {ModelName} [options]

Options are model-dependent and not documented here, and the models
and usage of this module is subject to change.

Models include::

    toy101
    toy102
    toy103
    drift
    periodic

tests.py
~~~~~~~~

Automated tests of various models and options.




Use as a library
----------------

Above, a command line interface is presented.  All code is modular an
can be imported and used directly from Python, without needing to
create temporary files.  This is the author's primary method of using
this program.

Unfortunately, this isn't documented yet (and the interface isn't
totally settled yet)




Input formats
-------------

Input is text files.  There is one row for each event.  One column
represents the time.  Optionally, one column can represent the weight
of each event.  All other columns specify the event.  Comments (lines
beginning with '#') are ignored.  Time and weight columns must be
numbers, but all other columns can be any string.

Use the "-t" option to specify the column with times (default: 0th
column), and use "-w" to specify the weight column (default:
unweighted).  NOTE: column indexes begin from zero!

Example 1: simple file.::

    #t event
    0 aaa
    0 bbb
    1 aaa
    2 ccc

Example 2: directed graph.  'a', 'b', 'c' are nodes.  To use an
undirected graph, use the "--unordered" option.::

    #t e1 e2
    0 a b
    0 a c
    1 c b
    2 a c

Example 2: Weighted graph.  Note the column order.  To read this, use
the options "-t 3 -w 2".  For a undirected graph, use the
"--unordered" option.::

    # e1 e2 weight time
    a b 1.0 0
    a .	1.0 0
    c b	2.0 1
    a c	1.0 2

GROUPED FORMAT: With the option "--grouped", you can have multiple
events on the same line.  Each event is one space-separated string.
Time lines can repeat.  Use "-t" to specify the time column, if not
the first, and "-w" to represent a weight column if it exists.  The
same weight applies to everything on the line.::

    # t events
    0 a b d
    1 a e f g h
    2 b c d




Output formats
--------------

The following files are written:

:OutputName.out.txt:
    Contains one row for each interval.  There is a comment at the top
    describing format.  Columns are:

      :tlow:          lower bound of segment  (closed, tlow<=segment<thigh)
      :thigh:         upper bound of segment  (open,   tlow<=segment<thigh)
      :dt:            length of interval
      :sim:           value of Jaccard score or other measure between this
                      interval and next
      :len(old_es):   Number of events in this interval
      :measure_data:  Information specific to the measure (like
                      Jaccard) being computer.  For Jaccard, there are
                      four values.  (intersection_size, union_size,
                      num_elements_left, num_elements_right)

    Note: first line has slightly different format, since it is the
    starting interval.

:OutputName.out.J.txt:
    Contains information on every unique minimization process.  There
    is one block for each segment interval, separated by blank lines.

      :t:    Time interval endpoint checked
      :val:  Jaccard (or other value) at this point.
      :dt:   Time interval checked
      :measure_data: same as above

:OutputName.out.J.{pdf,png}:
    Plots

