from setuptools import setup, find_packages

setup(
    name="hdc_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    author="Rahees Ahmed",
    author_email="raheesahmed256@gmail.com",
    description="A Python implementation of Hyperdimensional Computing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/raheesahmed/hdc_model",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
