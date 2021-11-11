import setuptools

with open("README.md", "r") as fd:
    long_description = fd.read()

setuptools.setup(
    name="chameleonclt",
    version="0.0.1",
    description="Python package of the clustering algorithm CHAMELEON",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lin Dong et al.",
    author_email="lindong941107@gmail.com",
    url="https://github.com/cskyan/chameleon_cluster",
    py_modules = ['chameleon', 'visualization'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
