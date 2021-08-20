# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import inspect
import traceback
import os

from typing import Any, List


def contract(condition: bool, format_string: str, *format_args: Any) -> None:
	"""
	Alternative to Python's built-in assert function.
	`contract()` also provides a human-readable error message, inspired by Rust's panic messages (https://play.rust-lang.org/?code=%23!%5Ballow(unused)%5D%0A%23!%5Ballow(unreachable_code)%5D%0Afn%20main()%20%7B%0Apanic!()%3B%0Apanic!(%22this%20is%20a%20terrible%20mistake!%22)%3B%0Apanic!(4)%3B%20%2F%2F%20panic%20with%20the%20value%20of%204%20to%20be%20collected%20elsewhere%0Apanic!(%22this%20is%20a%20%7B%7D%20%7Bmessage%7D%22%2C%20%22fancy%22%2C%20message%20%3D%20%22message%22)%3B%0A%7D&edition=2018).
	The first thing `contract()` does is check the given condition,
	thus it has very little overhead if the condition is met.
	If the condition is not met, `contract()` can have significant overhead due to formatting needs etc.,
	so it is important that you always check the "error" condition, and not the "ok" condition.

	:param condition:	The condition to be checked.
	:param format_string:	A format string that will be used to form a message.
	:param format_args:	Arguments to be filled in, in the `format_string`.
	:return:
	"""
	if not condition:
		formatted_msg: str = format_string.format(*format_args)

		stack_frame: inspect.FrameInfo = inspect.stack()[1]
		code = stack_frame[0].f_code

		filename: str = code.co_filename
		line: int = stack_frame.lineno

		concatenated_source_lines: str = ""

		try:
			source_lines: List[str] = inspect.getsourcelines(code)[0]

			relevant_source_lines: List[str] = source_lines[max(0, line - 2):min(len(source_lines), line + 2)]
			concatenated_source_lines = '\n'.join(relevant_source_lines)
		except OSError as oe:
			pass

		# Format the current stack trace, limiting formatting at 50 stack frames.
		formatted_stack_trace: str = ''.join(traceback.format_stack(limit=50)[:-1])

		error_msg: str = f"Process {os.getpid()} panicked at \"{formatted_msg}\", {filename}:{line}\n\n{concatenated_source_lines}\n{formatted_stack_trace}"

		raise AssertionError(error_msg)
