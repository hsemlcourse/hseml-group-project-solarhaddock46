.PHONY: lint test

lint:
	flake8 src/ tests/

test:
	pytest tests/test.py -v
