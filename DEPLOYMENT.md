# Deployment Guide for Proctor AI

This document outlines the deployment process and setup required for the proctor-ai package.

## GitHub Secrets Setup

To enable automatic deployment to PyPI, you need to set up the following secrets in your GitHub repository:

### Required Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

1. **`PYPI_API_TOKEN`** - For production PyPI deployment
   - Get this from: https://pypi.org/manage/account/token/
   - Create a new API token for the `proctor-ai` project
   - Use this for tagged releases (production deployment)

2. **`TEST_PYPI_API_TOKEN`** - For test PyPI deployment  
   - Get this from: https://test.pypi.org/manage/account/token/
   - Create a new API token for the test environment
   - Used for automatic deployment on main/master branch pushes

### Setting up PyPI API Tokens

1. **Production PyPI (pypi.org)**:
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Token name: `proctor-ai-github-actions`
   - Scope: `Entire account` or `Specific project: proctor-ai`
   - Copy the token (starts with `pypi-`)
   - Add to GitHub Secrets as `PYPI_API_TOKEN`

2. **Test PyPI (test.pypi.org)**:
   - Go to https://test.pypi.org/manage/account/token/
   - Click "Add API token"  
   - Token name: `proctor-ai-test-github-actions`
   - Scope: `Entire account`
   - Copy the token (starts with `pypi-`)
   - Add to GitHub Secrets as `TEST_PYPI_API_TOKEN`

## CI/CD Workflow

The GitHub Actions workflow (`.github/workflows/python-package.yml`) handles:

### On Pull Requests:
- ✅ Run tests across Python 3.8-3.11
- ✅ Code linting and formatting checks
- ✅ Core functionality tests

### On Push to main/master:
- ✅ All PR checks
- ✅ Build package
- ✅ Deploy to Test PyPI automatically
- 🔧 Allows testing the package before production release

### On Tagged Release (v*):
- ✅ All previous checks
- ✅ Build package
- ✅ Deploy to Production PyPI automatically
- 🚀 Makes the package available to users

## Release Process

### 1. Version Update
```bash
# Update version (choose one)
make version-bump-patch  # 0.1.1 → 0.1.2
make version-bump-minor  # 0.1.1 → 0.2.0  
make version-bump-major  # 0.1.1 → 1.0.0
```

### 2. Commit and Push
```bash
git add pyproject.toml
git commit -m "Bump version to X.Y.Z"
git push origin main
```

### 3. Create Release Tag
```bash
git tag v0.1.2  # Use your actual version
git push origin v0.1.2
```

### 4. Automated Deployment
GitHub Actions will automatically:
- Run all tests
- Build the package  
- Deploy to PyPI
- Create a GitHub release

## Manual Deployment

If needed, you can deploy manually using the Makefile:

```bash
# Test deployment
make deploy-test-permissive

# Production deployment  
make deploy-prod-permissive
```

## Package Information

- **Package Name**: `proctor-ai`
- **PyPI URL**: https://pypi.org/project/proctor-ai/
- **Installation**: `pip install proctor-ai`
- **Import**: `from proctor import ZeroShotCoT` (import path unchanged)

## Troubleshooting

### Common Issues:

1. **403 Forbidden during upload**:
   - Check API token permissions
   - Ensure token is for the correct environment (test vs prod)
   - Verify token hasn't expired

2. **Package name conflicts**:
   - Package name `proctor-ai` is unique and available
   - Source code still uses `proctor` module name internally

3. **Build failures**:
   - Run `make test-core` locally first
   - Check Python version compatibility
   - Ensure all dependencies are properly specified

### Getting Help:

- Check the GitHub Actions logs for detailed error messages
- Run the same commands locally using the Makefile
- Use `make help` to see all available commands