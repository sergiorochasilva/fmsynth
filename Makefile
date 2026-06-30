PYTHON ?= python

.PHONY: listen-big18
listen-big18:
	. ./.venv/bin/activate && $(PYTHON) listen_nsynth_pairs.py --pairs-file listen_big18_examples.txt
