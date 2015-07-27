This implements the dynamic segmentation method of [REFERENCE].

This method can take any type of timestamped data, where events have
some sort of distinct identities.  Events should repeat, and have
short-term similarity and long-term differences.  This method will
create segments which group time regions with similar events
together.  In other words, an intrinsic data scale is detected which
is not related to just the event rate.

* Our paper describing the method is at [REFERENCE].
* A user manual exists at doc/Manual.rst
* A detailed algorithmic description exists at doc/Algorithm.tex
