import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="konectipy",
    version="0.0.3",
    author="Akileshvar A",
    author_email="akileshvar008@gmail.com",
    description="A network analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AkiMosi/konectipy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
