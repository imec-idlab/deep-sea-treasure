# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import re
import setuptools
import sys

from typing import List


def get_version() -> str:
	try:
		from deep_sea_treasure.__version__ import VERSION

		return VERSION
	except ImportError as _:
		return "0.0.0"


def get_long_description() -> str:
	long_description: str = ""

	with open("README.md", "r") as readme_file:
		long_description = readme_file.read()

	return long_description


def get_python_version(mypy_config: str = ".gitlab/mypy.ini") -> str:
	"""
	Get Python version from mypy config, to prevent mypy version from diverging from package version.
	:return:
	"""
	if os.path.isfile(mypy_config):
		mypy_version = (0, 0)

		with open(mypy_config, "r") as mypy_config_file:
			lines: List[str] = mypy_config_file.readlines()

			# Check which version mypy assumes
			# We only support Python 3
			version_regex: re.Pattern = re.compile("^python_version\\s*=\\s*3\\.([0-9]+)$")

			version_lines: List[str] = [line for line in lines if version_regex.match(line)]

			# Just take the first line and roll with that, there should only be one of these in here anyway.
			mypy_version = (3, int(version_regex.match(version_lines[0]).group(1)))

		return '.'.join([str(v) for v in mypy_version])
	else:
		major, minor, _, _, _ = sys.version_info

		return f"{major}.{minor}"


def get_dependencies(requirements_filename: str = "requirements.txt") -> List[str]:
	dependencies: List[str] = []

	with open(requirements_filename, 'r') as requirements_file:
		dependencies = requirements_file.readlines()

	dependencies = [dep for dep in dependencies if not (dep.startswith('-') or dep.startswith('#'))]

	return dependencies


setuptools.setup(
		name="deep_sea_treasure",
		version=get_version(),
		author="Thomas Cassimon, Reinout Eyckerman",
		author_email="thomas.cassimon@uantwerpen.be,reinout.eyckerman@uantwerpen.be",
		description="An environment for testing multi-objective reinforcement learning algorithms.",
		long_description=get_long_description(),
		long_description_content_type="text/markdown",
		classifiers=[
			"Development Status :: 3 - Alpha",
			"Environment :: Console",
			"Intended Audience :: Science/Research",
			"License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
			"Natural Language :: English",
			"Operating System :: OS Independent",
			"Programming Language :: Python :: 3",
			"Topic :: Scientific/Engineering :: Artificial Intelligence"
		],
		packages=["deep_sea_treasure"],
		package_data={"deep_sea_treasure": ["res/*.png", "res/__init__.py", "schema/*.json", "schema/__init__.py", "py.typed"]},
		python_requires=f">={get_python_version()}",
		install_requires=get_dependencies()
)
