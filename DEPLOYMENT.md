# Deployment Guide

This document outlines the deployment and release process for the Proctor package.

## GitHub CI/CD Pipeline

We've set up a GitHub Actions workflow that automates testing, building, and deploying the package. The workflow is defined in `.github/workflows/python-package.yml`.

### Workflow Stages

1. **Test**: Runs tests on multiple Python versions (3.8, 3.9, 3.10, 3.11)
   - Installs dependencies
   - Runs linting with ruff
   - Runs tests with pytest

2. **Build**: Builds the package into a distributable format
   - Creates source distribution and wheel
   - Uploads artifacts for use in the publish stage

3. **Publish**: Publishes the package to PyPI (only on tag pushes)
   - Downloads the built artifacts
   - Uses twine to upload to PyPI

### Secrets Required

To enable automatic deployment to PyPI, you need to set up the following secrets in your GitHub repository:

1. `PYPI_USERNAME`: Your PyPI username
2. `PYPI_PASSWORD`: Your PyPI password or token (recommended)

## Manual Release Process

You can also perform a release manually using the included release script:

1. Make sure all your changes are committed
2. Run the release script to bump the version and update the changelog:
   ```bash
   python scripts/release.py [major|minor|patch]
   ```
3. Update the CHANGELOG.md file with details about the release
4. Commit and push the changes along with the new tag

## Release Checklist

Before releasing a new version, ensure the following:

1. All tests are passing
2. Documentation is up to date
3. CHANGELOG.md is updated with all notable changes
4. Version number is updated in `proctor/__init__.py`
5. Any API changes are backward compatible (or clearly documented as breaking changes)

## Post-Release Tasks

After a successful release:

1. Create a GitHub Release with release notes
2. Announce the new release in relevant channels
3. Start planning for the next release

## Troubleshooting

If the automated deployment fails:

1. Check the GitHub Actions logs for error details
2. Verify that all secrets are correctly configured
3. Try a manual deployment using twine:
   ```bash
   python -m build
   twine upload dist/*
   ```

## PyPI Package Management

To manage your package on PyPI:

1. Visit https://pypi.org/project/proctor/
2. Log in with your PyPI credentials
3. You can view download statistics, manage releases, and more