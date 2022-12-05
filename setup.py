from setuptools import setup, find_packages

with open('requirements.txt') as f:
requirements = f.read().splitlines()

with open("README.md", "r") as fh:
long_description = fh.read()

setuptools.setup(
name="simple_scheduler", 
version="0.0.1",
author="Donghyun Kwak",
author_email="donghyun.kwak@naver.com",
description="A simplest scheduler for automated data preprocessing and nlu model training.",
long_description=long_description,
long_description_content_type="text/markdown",
url="https://github.com/dhkwak/simple_scheduler",
packages=setuptools.find_packages(),
classifiers=[
"Programming Language :: Python :: 3",
"License :: OSI Approved :: MIT License",
"Operating System :: OS Independent",
],
python_requires='>=3.6',
install_requires=requirements,
)