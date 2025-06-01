PYTHON = python3
PACKAGE_NAME = proctor-ai
SOURCE_DIR = proctor
CLEANUP_DIRS = ~/.cache/$(SOURCE_DIR) __pycache__ .pytest_cache .tox .coverage .nox *.egg-info dist build $(SOURCE_DIR)/__pycache__ tests/__pycache__

.PHONY: lint test test-core test-cov check clean install install-dev sync-dev install-uv build check-dist ensure-twine deploy-check deploy-check-permissive deploy-test deploy-test-permissive deploy-prod deploy-prod-permissive deploy version-bump-patch version-bump-minor version-bump-major publish help all

lint: 
	@echo "Running linter and formatter (Ruff)..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff format $(SOURCE_DIR)/ tests/ examples/; \
		uv run ruff check $(SOURCE_DIR)/ tests/ examples/ --fix; \
	else \
		$(PYTHON) -m ruff format $(SOURCE_DIR)/ tests/ examples/; \
		$(PYTHON) -m ruff check $(SOURCE_DIR)/ tests/ examples/ --fix; \
	fi

test:
	@echo "Running tests with pytest..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest tests/ --maxfail=1 --disable-warnings -q; \
	else \
		$(PYTHON) -m pytest tests/ --maxfail=1 --disable-warnings -q; \
	fi

test-core:
	@echo "Running core tests (excluding KNN implementation)..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest tests/ --ignore=tests/test_knn_implementation.py --maxfail=1 --disable-warnings -q; \
	else \
		$(PYTHON) -m pytest tests/ --ignore=tests/test_knn_implementation.py --maxfail=1 --disable-warnings -q; \
	fi

test-cov:
	@echo "Running tests with coverage..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest tests/ --cov=$(SOURCE_DIR) --cov-report=html --cov-report=term-missing; \
	else \
		$(PYTHON) -m pytest tests/ --cov=$(SOURCE_DIR) --cov-report=html --cov-report=term-missing; \
	fi

check: 
	@echo "Checking code formatting and linting..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check $(SOURCE_DIR)/ tests/ examples/ --output-format=github; \
		uv run ruff format $(SOURCE_DIR)/ tests/ examples/ --check; \
	else \
		$(PYTHON) -m ruff check $(SOURCE_DIR)/ tests/ examples/ --output-format=github; \
		$(PYTHON) -m ruff format $(SOURCE_DIR)/ tests/ examples/ --check; \
	fi

clean:
	@echo "Cleaning up build artifacts and cache..."
	rm -rf $(CLEANUP_DIRS)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

install:
	@echo "Installing package in development mode..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for installation..."; \
		uv pip install -e .; \
	else \
		echo "Using pip for installation..."; \
		$(PYTHON) -m pip install -e .; \
	fi

install-dev:
	@echo "Installing package with development dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for installation..."; \
		uv sync --dev --all-extras; \
	else \
		echo "Using pip for installation..."; \
		$(PYTHON) -m pip install -e ".[dev,all]"; \
	fi

sync-dev:
	@echo "Syncing development dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --dev --all-extras; \
		uv run pip list | grep -E "(ruff|pytest|twine)"; \
	else \
		$(PYTHON) -m pip install -e ".[dev,all]"; \
	fi

install-uv:
	@echo "Installing with uv..."
	uv sync --dev

build:
	@echo "Building package..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for building..."; \
		uv build; \
	else \
		echo "Using python build..."; \
		$(PYTHON) -m build; \
	fi

check-dist:
	@echo "Checking distribution files..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run twine check dist/*; \
	else \
		$(PYTHON) -m twine check dist/*; \
	fi

ensure-twine:
	@echo "Ensuring twine is available..."
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --dev --all-extras; \
	else \
		$(PYTHON) -m pip install twine; \
	fi

deploy-check: clean sync-dev lint check test ensure-twine
	@echo "Running pre-deployment checks..."
	@echo "✓ Dependencies installed"
	@echo "✓ Code formatted and linted"
	@echo "✓ All tests passed"
	@echo "✓ Deployment tools ready"
	@echo "Ready for deployment!"

deploy-check-permissive: clean sync-dev check test-core ensure-twine
	@echo "Running pre-deployment checks (permissive mode)..."
	@echo "✓ Dependencies installed"
	@echo "✓ Code style checked (warnings allowed)"
	@echo "✓ Core tests passed (optional features skipped)"
	@echo "✓ Deployment tools ready"
	@echo "Ready for deployment (with style warnings)!"

deploy-test: deploy-check build check-dist
	@echo "Publishing to Test PyPI..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run twine upload --repository testpypi dist/*; \
	else \
		$(PYTHON) -m twine upload --repository testpypi dist/*; \
	fi
	@echo "✓ Package uploaded to Test PyPI"
	@echo "Test with: pip install -i https://test.pypi.org/simple/ proctor-ai"

deploy-test-permissive: deploy-check-permissive build check-dist
	@echo "Publishing to Test PyPI (permissive mode)..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run twine upload --repository testpypi dist/*; \
	else \
		$(PYTHON) -m twine upload --repository testpypi dist/*; \
	fi
	@echo "✓ Package uploaded to Test PyPI"
	@echo "Test with: pip install -i https://test.pypi.org/simple/ proctor-ai"

deploy-prod: deploy-check build check-dist
	@echo "🚀 Deploying to Production PyPI..."
	@read -p "Are you sure you want to publish to PyPI? (y/N): " confirm && [ "$$confirm" = "y" ]
	@if command -v uv >/dev/null 2>&1; then \
		uv run twine upload dist/*; \
	else \
		$(PYTHON) -m twine upload dist/*; \
	fi
	@echo "✅ Package successfully deployed to PyPI!"
	@echo "Install with: pip install proctor-ai"

deploy-prod-permissive: deploy-check-permissive build check-dist
	@echo "🚀 Deploying to Production PyPI (permissive mode)..."
	@read -p "Are you sure you want to publish to PyPI? (y/N): " confirm && [ "$$confirm" = "y" ]
	@if command -v uv >/dev/null 2>&1; then \
		uv run twine upload dist/*; \
	else \
		$(PYTHON) -m twine upload dist/*; \
	fi
	@echo "✅ Package successfully deployed to PyPI!"
	@echo "Install with: pip install proctor-ai"

deploy: deploy-prod

version-bump-patch:
	@echo "Bumping patch version..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -c "import re; \
		content = open('pyproject.toml').read(); \
		version = re.search(r'version = \"([^\"]+)\"', content).group(1); \
		major, minor, patch = version.split('.'); \
		new_version = f'{major}.{minor}.{int(patch)+1}'; \
		new_content = re.sub(r'version = \"[^\"]+\"', f'version = \"{new_version}\"', content); \
		open('pyproject.toml', 'w').write(new_content); \
		print(f'Version bumped to {new_version}')"; \
	else \
		$(PYTHON) -c "import re; \
		content = open('pyproject.toml').read(); \
		version = re.search(r'version = \"([^\"]+)\"', content).group(1); \
		major, minor, patch = version.split('.'); \
		new_version = f'{major}.{minor}.{int(patch)+1}'; \
		new_content = re.sub(r'version = \"[^\"]+\"', f'version = \"{new_version}\"', content); \
		open('pyproject.toml', 'w').write(new_content); \
		print(f'Version bumped to {new_version}')"; \
	fi

version-bump-minor:
	@echo "Bumping minor version..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -c "import re; \
		content = open('pyproject.toml').read(); \
		version = re.search(r'version = \"([^\"]+)\"', content).group(1); \
		major, minor, patch = version.split('.'); \
		new_version = f'{major}.{int(minor)+1}.0'; \
		new_content = re.sub(r'version = \"[^\"]+\"', f'version = \"{new_version}\"', content); \
		open('pyproject.toml', 'w').write(new_content); \
		print(f'Version bumped to {new_version}')"; \
	else \
		$(PYTHON) -c "import re; \
		content = open('pyproject.toml').read(); \
		version = re.search(r'version = \"([^\"]+)\"', content).group(1); \
		major, minor, patch = version.split('.'); \
		new_version = f'{major}.{int(minor)+1}.0'; \
		new_content = re.sub(r'version = \"[^\"]+\"', f'version = \"{new_version}\"', content); \
		open('pyproject.toml', 'w').write(new_content); \
		print(f'Version bumped to {new_version}')"; \
	fi

version-bump-major:
	@echo "Bumping major version..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -c "import re; \
		content = open('pyproject.toml').read(); \
		version = re.search(r'version = \"([^\"]+)\"', content).group(1); \
		major, minor, patch = version.split('.'); \
		new_version = f'{int(major)+1}.0.0'; \
		new_content = re.sub(r'version = \"[^\"]+\"', f'version = \"{new_version}\"', content); \
		open('pyproject.toml', 'w').write(new_content); \
		print(f'Version bumped to {new_version}')"; \
	else \
		$(PYTHON) -c "import re; \
		content = open('pyproject.toml').read(); \
		version = re.search(r'version = \"([^\"]+)\"', content).group(1); \
		major, minor, patch = version.split('.'); \
		new_version = f'{int(major)+1}.0.0'; \
		new_content = re.sub(r'version = \"[^\"]+\"', f'version = \"{new_version}\"', content); \
		open('pyproject.toml', 'w').write(new_content); \
		print(f'Version bumped to {new_version}')"; \
	fi

publish: deploy-prod

help:
	@echo "Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  lint         - Format and fix code with ruff"
	@echo "  test         - Run all tests with pytest"
	@echo "  test-core    - Run core tests (excluding optional KNN features)"
	@echo "  test-cov     - Run tests with coverage report"
	@echo "  check        - Check code formatting and linting without fixing"
	@echo "  clean        - Remove build artifacts and cache files"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  sync-dev     - Sync all dev dependencies (recommended for uv)"
	@echo "  install-uv   - Install with uv (if available)"
	@echo "  all          - Run lint, test, and clean"
	@echo ""
	@echo "Deployment:"
	@echo "  build                   - Build package for distribution"
	@echo "  check-dist              - Validate built distribution files"
	@echo "  deploy-check            - Run all pre-deployment checks (strict)"
	@echo "  deploy-check-permissive - Run pre-deployment checks (allows style warnings)"
	@echo "  deploy-test             - Deploy to Test PyPI (strict checks)"
	@echo "  deploy-test-permissive  - Deploy to Test PyPI (permissive checks)"
	@echo "  deploy-prod             - Deploy to Production PyPI (strict checks)"
	@echo "  deploy-prod-permissive  - Deploy to Production PyPI (permissive checks)"
	@echo "  deploy                  - Alias for deploy-prod"
	@echo "  publish                 - Alias for deploy-prod"
	@echo ""
	@echo "Version Management:"
	@echo "  version-bump-patch - Bump patch version (0.1.0 -> 0.1.1)"
	@echo "  version-bump-minor - Bump minor version (0.1.0 -> 0.2.0)"
	@echo "  version-bump-major - Bump major version (0.1.0 -> 1.0.0)"
	@echo ""
	@echo "  help         - Show this help message"

all: lint test clean
	@echo "All development tasks completed!" 