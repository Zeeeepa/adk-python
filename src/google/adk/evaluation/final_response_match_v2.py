# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import re

from typing_extensions import override

from ..models.llm_response import LlmResponse
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .evaluator import EvaluationResult
from .evaluator import PerInvocationResult
from .response_auto_rater import get_eval_status
from .response_auto_rater import get_text_from_content
from .response_auto_rater import ResponseAutoRater


class ResponseMatchV2Evaluator(ResponseAutoRater):
  """AutoRater-based evaluator to judge final response."""

  def __init__(
      self,
      eval_metric: EvalMetric,
      auto_rater_prompt_template: str,
      multi_turn: bool = False,
  ):
    super().__init__(eval_metric, auto_rater_prompt_template)
    self._multi_turn = multi_turn

  @override
  def format_auto_rater_prompt(
      self, actual_invocation: Invocation, expected_invocation: Invocation
  ) -> str:
    reference = get_text_from_content(expected_invocation.final_response)
    response = get_text_from_content(actual_invocation.final_response)
    user_prompt = get_text_from_content(expected_invocation.user_content)
    return self._auto_rater_prompt_template.format(
        function_api_spec="None",
        prompt=user_prompt,
        response=response,
        golden_response=reference,
    )

  @override
  def convert_auto_rater_response_to_score(
      self, llm_response: LlmResponse
  ) -> float:
    try:
      response_text = get_text_from_content(llm_response.content).strip()
      match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
      if match:
        response_json_text = match.group(1)
        parsed_response = json.loads(response_json_text)
      else:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError as e:
      raise ValueError(
          f"Failed to parse auto rater response: {llm_response}"
      ) from e
    is_valid = parsed_response["is_the_agent_response_valid"].lower() == "valid"
    return 1.0 if is_valid else 0.0

  @override
  def aggregate_invocation_results(
      self, per_invocation_results: list[PerInvocationResult]
  ) -> EvaluationResult:
    """Computes the fraction of invocation results that are valid."""
    num_valid = 0
    for result in per_invocation_results:
      if result.score == 1.0:
        num_valid += 1
    overall_score = num_valid / len(per_invocation_results)
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=get_eval_status(
            overall_score, self._eval_metric.threshold
        ),
        per_invocation_results=per_invocation_results,
    )
