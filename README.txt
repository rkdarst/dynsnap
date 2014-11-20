Dynamic Snapshot tool
=====================

This tool dynamically finds snapshots in data.

It is still under development, so things are disorganized and
incomplete.



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

CACHING: You can use "--cache" to cache the processed input file  in
a '.cache' file beside the original input file.  This is an sqlite
database, but that is unimportant.  The cache depends on the "-t",
"-w", and "--unordered" options.  If you use "--cache" and change any
of these options, you MUST run with the "--regen" option to regenerate
the cache.


Usage
=====

There are no dependencies, and only the `dynsnap.py` file is
required.  Command line syntax is:
   python dynsnap.py {InputFile} {OutputName} [options]

Arguments:

InputFile
    Raw input file.
OutputName
    Prefix for output files.  Output is written to OutputName.out.txt,
    etc.

Options:

-t N
    Specify column holding time.   Default: 0
-w N
    Specify column holding weight: Default: unweighted
--unordered
    If given, the ordering of other columns does not matter, and lines
    with events "aaa bbb ccc" and "aaa ccc bbb" are considered the
    same.
--grouped
    Alternative input format where each line can have multiple
    events.
--dont-merge-first
    Do not perform the "merge first two intervals" process.

--plot, -p
    Write some plots at OutputName.{pdf,png}.  Requires matplotlib and
    pcd.

--cache
    Store the processed input file in InputFile.cache.  This will
    speed up re-runs of the method.
--regen
    Regenerate data in the cache (remove it before starting, then
    recreate it.)

--tstart
--dtmin
--dtmax
--dtstep
--dtextra
    Undocumented internal options.



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

