
.PHONY: compilePDF clean

_default:
	make compilePDF

setup: make_plots
	@#;

make_plots:
	@$(MAKE) plots -C ../data/

compilePDF: make_plots
	mkdir -p outputFiles/targetT1/
	# mkdir -p outputFiles/externalPlots/
	@latexmk -pdf -pvc -view=none -shell-escape -auxdir=outputFiles/targetT1/ -outdir=../ -interaction=nonstopmode ./27086534RW741assignment*.tex
# compilePDFTask2:
# 	mkdir -p outputFiles/targetT2/
# 	@latexmk -pdf -pvc -view=none -shell-escape -auxdir=outputFiles/targetT1/ -outdir=./ -interaction=nonstopmode report.tex
clean:
	rm -rf outputFiles/target
