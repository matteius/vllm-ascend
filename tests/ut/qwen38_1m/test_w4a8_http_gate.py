# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tools.qwen4exp.evaluate_w4a8_http import score_answer


@pytest.mark.parametrize("content", [None, "", "A or B", "The answer is A", "<think>A</think>", "AB", "a"])
def test_ambiguous_or_missing_output_cannot_pass(content):
    assert score_answer(content, "A") == (None, False)


def test_single_letter_answer_scoring():
    assert score_answer(" A.\n", "A") == ("A", True)
    assert score_answer("B", "A") == ("B", False)
