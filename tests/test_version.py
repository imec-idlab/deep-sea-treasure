# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import unittest

import deep_sea_treasure


class VersionTest(unittest.TestCase):
	def test_build_no_injection(self):
		# We are running in a CI Pipeline
		self.assertRegex(deep_sea_treasure.__version__.VERSION, "\\d+\\.\\d+\\.\\d+")

	def test_commit_injection(self):
		# We are running in a CI Pipeline
		if "CI_PIPELINE_ID" in os.environ:
			self.assertRegex(deep_sea_treasure.__version__.COMMIT_HASH, "^[0-9a-fA-F]{40}$")
		else:
			self.assertRegex(deep_sea_treasure.__version__.COMMIT_HASH, "^.*$")
