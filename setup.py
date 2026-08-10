"""
MemoryTrack setup configuration
"""
from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="memorytrack",
    version="1.0.0",
    author="MemoryTrack Team",
    author_email="memorytrack@example.com",
    description="Adaptive Memory-Based Multi-Camera Missing Person Tracking System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chinthakuntaharini/MemoryTrack",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "gpu": ["faiss-gpu>=1.7.0"],
        "enhanced": [
            "bytetrack>=1.0.0",
            "torchreid>=0.2.5",
            "mmpose>=1.0.0",
            "mmdet>=3.0.0",
            "mmcv>=2.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memorytrack=main:main",
            "memorytrack-dashboard=dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "database/*.sql", "*.md"],
    },
    zip_safe=False,
    keywords="computer-vision person-tracking missing-person multi-camera surveillance ai",
    project_urls={
        "Bug Reports": "https://github.com/chinthakuntaharini/MemoryTrack/issues",
        "Source": "https://github.com/chinthakuntaharini/MemoryTrack",
        "Documentation": "https://github.com/chinthakuntaharini/MemoryTrack/blob/main/README.md",
    },
)