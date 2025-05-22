from setuptools import find_packages, setup

setup(
    name="rdtattoo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.4",
        "pandas>=2.2.3",
        "matplotlib>=3.10.1",
        "plotly>=6.0.1",
        "scipy>=1.15.2",
        "streamlit>=1.44.1",
        "ipykernel>=6.29.5",
        "jupyter>=1.0.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="RD Tattoo Analysis and Visualization Package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
