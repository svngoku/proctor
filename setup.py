"""
Setup script for the proctor package.
"""
from setuptools import setup, find_packages

# Read version from package
about = {}
with open("proctor/__init__.py") as f:
    exec(next(filter(lambda line: line.startswith("__version__"), f)), about)

setup(
    name="proctor",
    version=about["__version__"],
    description="A comprehensive package for text-based prompting techniques",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/svngoku/proctor",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        "litellm>=1.0.0",
        "openai>=1.0.0",
        "python-dotenv>=0.20.0",
        "rich>=13.0.0"
    ],
    extras_require={
        "knn": [
            "sentence-transformers>=2.0.0",
            "scikit-learn>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.0.270",
            "build>=0.10.0",
        ],
        "all": [
            "sentence-transformers>=2.0.0",
            "scikit-learn>=1.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.0.270",
            "build>=0.10.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)