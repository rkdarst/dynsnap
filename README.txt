Dynamic Snapshot tool
=====================

This tool dynamically finds snapshots in data.

It is still under development, so things are disorganized and
incomplete.




Basic principles
================

This code takes a list of events (each event occurring at a certain
time), and segments the events into time intervals.

The first input is some InputFile.  The most basic format has one
column for time, and all remaining columns are interpreted as strings,
and specify the event.

For efficiency reasons, the input file can be compiled into a sqlite
database using ``events.py``.  This allows optimized lookups on all
events, and ability to quickly re-run with different options.  Even if
the temporary database is not made with ``events.py``, it is made in
memory on every run, anyway.

Finally, one would use ``dynsnap.py`` to run on the cached data (or
you could run directly on input files, if you wanted), and produce
some text output files, and optionally plot files.  These would need
to be analyzed by yourself to understand what is going on.




Quick example
=============

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

Now, we process this into the optimized sqlite database:

    python events.py input.txt input.sqlite -t 0 -w 1

``input.txt`` and ``input.sqlite`` names are input/output filenames.
``-t 0`` indicates that time is in the zeroth column.  ``-w 1``
indicates that the first (zero-indexed) column contains weights.
These weights are recorded within the database, but to use them you
need the ``-w -1`` option in the next step.  Weights default to 1 if
``-w`` is not specified.

Then, we can run the actual analysis:

    python dynsnap.py input.sqlite input.out -p -w -1

The ``-p`` option indicates that we want to create some plots, in
addition to the standard output files.  You must include the ``-w``
option here to get the weighted analysis, but the column number is
irrelevant since weights are already stored in the database (so ``-1``
is used).

Several output files are written, starting with ``input.out``.

input.out.txt:
    Contains columns ``t_low t_high interval_length similarity_value ...``
input.out.J.txt:
    Contains information on the maximization process.
input.out.{pdf, png}:
    Plots of the process.




Command line usage
==================

dynsnap.py - finding snapshot intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This program does the actual analysis.  The most efficient method is
to first use ``events.py`` to create a database of events, and run
this program on that database.  However, you can run directly on input
.txt files, too, and for that some options of ``events.py`` are
duplicated here.

There are very many options for controlling this process.  While a
description of options is given here, details are in the accompanying
scientific README.


Basic syntax:

   python dynsnap.py {InputFile} {OutputName} [options]

Arguments:

InputFile
    Raw input file, or processed .sqlite input file.
OutputName
    Prefix for output files.  Output is written to OutputName.out.txt,
    etc.

Options:

-w N
    Specify weighted analysis.  If operating on raw input, the
    zero-indexed column number of the weights in the file.  If
    operating on an sqlite database, specify ``-1`` or anything to
    turn on weighted analysis.
-t N
--unordered
--grouped
    See documentation for events.py below.

--dont-merge-first
    Do not perform the "merge first two intervals" process.

--plot, -p
    Write some plots at OutputName.{pdf,png}.  Requires matplotlib and
    pcd.

--tstart T
    Start analysis only after this time.
--tstop T
    Stop analysis after this time.
--dtmode=NAME
    Select among the three types of search patterns: ``linear``,
    ``log``, and ``event``.
    linear: Simple, dumb linear search.  Set --dtstep=STEP_SIZE to
            adjust scale, and optionally --dtmin=, --dtmax=, and
            --dtextra= to control other parameters of the search.
    event: Scans exactly the dt intervals corresponding to the next
           events.  This adapts to the scale of the data, but is still
           inefficient if the optimal time scale is much larger than
           the  inter-event time.
    log: Logarithmic scanning.  Scans 1, 2, .. 99, 100, 110, 120,
         ... 990, 1000, 1100, ....  The first interval is adapted to
         the proper size of the problem.  The base of 10 is fixed, and
         this is not very configurable, but should adapt to most
         problem scales.
    The default is ``log``, which should have good performance on most
    problems.

--peakfinder=NAME
    Method of finding peaks of J, if there is a plateau of the same
    value.  Options are longest, shortest, greedy.
    longest: longest-time plateau value.
    shortest: shortest-time plateau value.
    greedy: A greedy search for longest plateau value.  As soon as the
            first decrease is detected, abort the search and use the
            longest plateau value.  This is in contrast to ``longest``
            and ``shortest``, which go a bit further and make sure
            that there is no future greater maximum.
    The default is ``longest``


Options for --dtmode=linear
--dtstep
    Set the scale for searching.
--dtmin
--dtmax
--dtstep
--dtextra
    Undocumented internal options.



events.py -- preprocess input files into an optimized database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This program will compile input files into optimized sqlite databases.
The primary use of this is to avoid the parsing step every time you
re-run the analysis with slightly different options, leading to much
greater efficiency.  This converts all events into integers and loses
original event names/ids.


Basic syntax:

   python events.py {InputFile} {OutputName} [options]

Arguments:

InputFile
    Raw input file
OutputName
    Output sqlite database.

-t N
    Specify column holding time.   Default: 0
-w N
    Specify column holding weight: A value of -1 means that there is
    no weight column, but calculate using weighted mechanics anyway.
    Default: unweighted.
--unordered
    If given, the ordering of other columns does not matter, and lines
    with events "aaa bbb ccc" and "aaa ccc bbb" are considered the
    same.
--grouped
    Alternative input format where each line can have multiple
    events.



models.py
~~~~~~~~~

This is an interface to various toy models.  Run the program with a
name of a model to generate output.  Various model options can be set
via command line.  The ``--grouped`` option can be given to make
output in grouped format (see below).

Models include::

    toy101
    toy102
    toy103
    drift
    periodic

tests.py
~~~~~~~~

Automated tests of various models and options.




Input formats
=============

Input is text files.  There is one row for each event.  One column
represents the time.  Optionally, one column can represent the weight
of each event.  All other columns specify the event.  Comments (lines
beginning with '#') are ignored.  Time and weight columns must be
numbers, but all other columns can be any string.

Use the "-t" option to specify the column with times (default: 0th
column), and use "-w" to specify the weight column (default:
unweighted).  NOTE: column indexes begin from zero!

Example 1: simple file
    #t event
    0 aaa
    0 bbb
    1 aaa
    2 ccc

Example 2: directed graph.  'a', 'b', 'c' are nodes.  To use an
undirected graph, use the "--unordered" option.
    #t e1 e2
    0 a b
    0 a c
    1 c b
    2 a c

Example 2: Weighted graph.  Note the column order.  To read this, use
the options "-t 3 -w 2".  For a undirected graph, use the
"--unordered" option.
    # e1 e2 weight time
    a b 1.0 0
    a .	1.0 0
    c b	2.0 1
    a c	1.0 2

GROUPED FORMAT: With the option "--grouped", you can have multiple
events on the same line.  Each event is one space-separated string.
Time lines can repeat.  Use "-t" to specify the time column, if not
the first, and "-w" to represent a weight column if it exists.  The
same weight applies to everything on the line.
    # t events
    0 a b d
    1 a e f g h
    2 b c d




Output formats
==============

The following files are written:

OutputName.out.txt
    Contains one row for each interval.  There is a comment at the top
    describing format.  Columns are:
      tlow:          lower bound of snapshot  (closed, tlow<=snapshot<thigh)
      thigh:         upper bound of snapshot  (open,   tlow<=snapshot<thigh)
      dt:            length of interval
      val:           value of Jaccard score or other measure between this
                     interval and next

      len(old_es):   Number of events in this interval
      measure_data:  Information specific to the measure (like
                     Jaccard) being computer.  For Jaccard, there are
                     four values.  (intersection_size, union_size,
                     num_elements_left, num_elements_right)

    Note: first line has slightly different format, since it is the
    starting interval.

OutputName.out.J.txt
    Contains information on every unique minimization process.  There
    is one block for each snapshot interval, separated by blank lines.

      t:    Time interval endpoint checked
      val:  Jaccard (or other value) at this point.
      dt:   Time interval checked
      measure_data: same as above

OutputName.out.J.{pdf,png}
    Plots

