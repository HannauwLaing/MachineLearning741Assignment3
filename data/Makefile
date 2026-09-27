
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval .PHONY: $(ARGS))
$(eval $(ARGS):;@:)

run:
	@mkdir -p ../report/outputFiles
	@python3 -m venv .venv > /dev/null
	# @.venv/bin/python3 ./MLAssignment3.py --skip-baseline
	@.venv/bin/python3 ./MLAssignment3.py $(ARGS)

run_attack:
	@mkdir -p ../report/outputFiles
	@python3 -m venv .venv > /dev/null
	# @.venv/bin/python3 ./MLAssignment3.py --skip-baseline
	@.venv/bin/python3 ./MLAssignment3.py --attack-priority $(ARGS)

plots:
	@mkdir -p ../report/outputFiles
	@python3 -m venv .venv > /dev/null
	@.venv/bin/python3 ./plot_assignment3_results.py

setup:
	@.venv/bin/pip install --upgrade pip > /dev/null
	@.venv/bin/pip install pandas > /dev/null
	@.venv/bin/pip install Jinja2 > /dev/null
	@.venv/bin/pip install matplotlib > /dev/null
	@.venv/bin/pip install tqdm > /dev/null
	@.venv/bin/pip install optuna > /dev/null
	@.venv/bin/pip install scikit-learn > /dev/null # Classification 
	@.venv/bin/pip install torch > /dev/null

clean:
	rm -rf .venv
	rm -rf data/.venv
