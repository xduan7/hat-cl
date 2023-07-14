help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    test:        unit test all test_*.py or *_test.py files"
	@echo "    coverage:    generate coverage report"
	@echo "    format:      check the format"
	@echo "    lint:        check the lint"
	@echo "    type:        check the type hints"
	@echo "    doc:         check the docstring"
	@echo "    check:       run format, lint, type, doc"
	@echo "    publish:     publish to pypi"

test:
	@echo ">>> Unit testing with pytest ..."
	@pytest || true
	@echo ""

coverage:
	@echo ">>> Unit testing with pytest ..."
	@coverage run -m pytest || true
	@echo ">>> Generating coverage report ..."
	@coverage report -m || true
	@echo ""

type:
	@echo ">>> Checking typing with mypy ..."
	@mypy hat || true
	@echo ""

format:
	@echo ">>> Sorting imports with isort ..."
	@isort . || true
	@echo ">>> Formatting with black ...."
	@black . || true
	@echo ""

lint:
	@echo ">>> Linting with flake8 ...."
	@flake8 hat || true
	@echo ">>> Checking unused code with vulture ...."
	@vulture --min-confidence 100 hat || true
	@echo ""

doc:
	@echo ">>> Check docstring with interrogate ...."
	@interrogate hat || interrogate hat -v || true
	@echo ""

check:
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) type
	@$(MAKE) doc

publish:
	@poetry build
	@poetry publish
