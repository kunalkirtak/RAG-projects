"""Packaging metadata for `pip install -e .` (editable/local install)."""

from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multi-document-research-agent",
    version="1.0.0",
    description="Enterprise-grade multi-document RAG agent with hybrid search, powered by Gemini.",
    author="Your Name",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "research-agent=main:main",
        ],
    },
)
