.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

CXX = g++
CXXFLAGS = -Wno-deprecated -D LINUX
SOURCES = xrcap/rootCrownSegmentation.cpp
LIBS = -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lboost_system -lboost_filesystem -lboost_program_options -ltbb

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 xrcap tests

test: ## run tests quickly with the default Python
	python setup.py test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source xrcap setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/xrcap.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ xrcap
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	rm -rvf xrcap/lib
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	if [ ! -d "xrcap/lib" ]; then mkdir -pv xrcap/lib; fi
	$(CXX) $(CXXFLAGS) $(SOURCES) $(DEPS) $(LIBS) -o xrcap/lib/rootCrownSegmentation
	sed -i "s/GIT_COMMIT = .*/GIT_COMMIT = '$(shell git rev-parse --short HEAD)'/g" xrcap/cli.py

	mkdir -pv xrcap/lib
	mkdir -pv /var/log/xrcap/batch_segmentation /var/log/xrcap/batch_skeleton /var/log/xrcap/qc_binary_images /var/log/xrcap/qc_point_clouds /var/log/xrcap/rootCrownImageAnalysis3D
	chmod -Rv 2777 /var/log/xrcap

	/usr/bin/env python3 -m pip install .

uninstall: clean ## remove package
	/usr/bin/env python3 -m pip uninstall -y xrcap
