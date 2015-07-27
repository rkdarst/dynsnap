all: Manual.html Algorithm.pdf

Manual.html: Manual.rst
	rst2html Manual.rst Manual.html

Algorithm.pdf: Algorithm.tex
	pdflatex Algorithm.tex
	pdflatex Algorithm.tex
