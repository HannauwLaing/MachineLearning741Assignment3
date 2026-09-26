

all: runData runReport
	@echo "Done"

runData:
	@$(MAKE) run -C ./data/

runReport: 
	@$(MAKE) -C ./report/

