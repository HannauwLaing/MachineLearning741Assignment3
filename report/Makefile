
.PHONY: compilePDF clean



compilePDF: plots
	mkdir -p outputFiles/targetT1/
	latexmk -pdf -shell-escape -auxdir=outputFiles/targetT1/ -outdir=../ -interaction=nonstopmode ./27086534RW741assignment*.tex


plots: 
	@$(MAKE) setup -C ../data/
	@$(MAKE) plots -C ../data/




watch: plots
	mkdir -p $(AUXDIR)
	latexmk -pdf -pvc -view=none -shell-escape -auxdir=outputFiles/targetT1/ -outdir=../ ./27086534RW741assignment*.tex
# compilePDF: plots
# 	mkdir -p outputFiles/targetT1/
# 	# mkdir -p outputFiles/externalPlots/
# 	@latexmk -pdf -pvc -view=none -shell-escape -auxdir=outputFiles/targetT1/ -outdir=../ -interaction=nonstopmode ./27086534RW741assignment*.tex
#

# compilePDFTask2:
# 	mkdir -p outputFiles/targetT2/
# 	@latexmk -pdf -pvc -view=none -shell-escape -auxdir=outputFiles/targetT1/ -outdir=./ -interaction=nonstopmode report.tex
clean:
	rm -rf outputFiles/target
