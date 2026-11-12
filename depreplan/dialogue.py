"""Run technique selection, initial planning, and feedback-guided replanning."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from .contracts import CANONICAL_TECHNIQUES, REVISION_MARKER, validate_record, validate_target
from .model import DialogueModel


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate model output key: {key}")
        value[key] = item
    return value


def _nonfinite(value):
    raise ValueError(f"Non-finite number: {value}")


def _assessment_text(assessment):
    if not isinstance(assessment, dict) or set(assessment) != {"scale_name", "origin", "total_score"}:
        raise ValueError("assessment requires scale_name, total_score and origin")
    score = assessment["total_score"]
    if not isinstance(assessment["origin"], str) or not assessment["origin"].strip():
        raise ValueError("assessment origin is required")
    allowed_types = (int, float) if assessment["origin"] == "model_estimate" else (int,)
    if (assessment["scale_name"] != "PHQ-8" or type(score) not in allowed_types
            or not 0 <= score <= 24 or not math.isfinite(score)):
        raise ValueError("PHQ-8 requires a total from 0 to 24; only a model estimate can be fractional")
    return "\n\nAssessment (marked provenance; not a clinical diagnosis):\n" + json.dumps(
        assessment, ensure_ascii=False, sort_keys=True, allow_nan=False)


class DialogueSession:
    """A planner receives the optional assessment; the executor sees the plan and history."""
    def __init__(self, generate, *, intake_text, problem_text, history, language,
                 assessment=None, planner_intake_text=None, sample_id="dialogue"):
        if not all(isinstance(x, str) and x.strip() for x in (sample_id, intake_text, problem_text)):
            raise ValueError("sample_id, intake_text and problem_text must be nonempty strings")
        if language not in ("en", "zh"):
            raise ValueError("language must be en or zh")
        if (not isinstance(history, list) or not history
                or any(not isinstance(m, dict) or m.get("role") not in ("user", "assistant")
                       or not isinstance(m.get("content"), str) or not m["content"].strip() for m in history)
                or history[-1]["role"] != "user"):
            raise ValueError("the opening history must end with a user message")
        if planner_intake_text is not None and assessment is not None:
            raise ValueError("supply either a prepared planner intake or a structured assessment")
        self.generate = generate
        self.sample_id = sample_id
        self.language = language
        self.problem = problem_text
        self.background = intake_text
        self.planner_intake = planner_intake_text if planner_intake_text is not None else intake_text
        if assessment is not None:
            self.planner_intake += _assessment_text(assessment)
        self.history = deepcopy(history)
        self.technique = None
        self.plan = None

    @classmethod
    def from_case(cls, generate, case):
        """Accept a public case or the candidate input portion of a DepGuide scenario."""
        if "conditions" in case:
            conditions = case["conditions"]
            plain = conditions["without_assessment"]
            selected = conditions.get("with_assessment") or plain
            return cls(generate, language=case["language"], sample_id=case.get("pack_id", "dialogue"),
                       intake_text=plain["intake_text"], planner_intake_text=selected["intake_text"],
                       problem_text=selected["problem_text"], history=selected["history"])
        return cls(generate, **case)

    def _record(self, task, intake, history, technique=None, plan=None):
        context = {"intake_text": intake, "problem_text": self.problem, "technique": technique,
                   "plan_text": plan, "response_strategy": None, "b_observations": []}
        if task == "c_method_selection":
            context.update(problem_text_status="native_present", allowed_techniques=list(CANONICAL_TECHNIQUES))
        return {"sample_id": self.sample_id, "task": task, "language": self.language,
                "context": context, "history": deepcopy(history)}

    def _call(self, record):
        validate_record(record, require_target=False)
        raw = self.generate(record)
        output = raw if record["task"] == "d_reply" else json.loads(
            raw, object_pairs_hook=_unique_keys, parse_constant=_nonfinite)
        validate_target(record["task"], output, record["context"])
        return output

    def _plan_and_reply(self, intake, history, technique, *, comparison_plan=None):
        planning = self._record("c_session_plan", intake, history, technique)
        plan = self._call(planning)["plan_text"]
        execution = self._record("d_reply", self.background, history, technique, plan)
        reply = self._call(execution)
        result = {"technique": technique, "plan": plan, "reply": reply}
        if comparison_plan is not None:
            original_reply = reply
            if comparison_plan != plan:
                original = deepcopy(execution)
                original["context"]["plan_text"] = comparison_plan
                original_reply = self._call(original)
            result = {"technique": technique, "history": deepcopy(history),
                      "original_plan": comparison_plan, "original_reply": original_reply,
                      "revised_plan": plan, "revised_reply": reply, "identical_replies": original_reply == reply}
        self.technique, self.plan = technique, plan
        self.history = [*history, {"role": "assistant", "content": reply}]
        return result

    def start(self):
        if self.plan is not None:
            raise ValueError("the first reply has already been generated")
        opening = "\n".join(("Client: " if m["role"] == "user" else "Assistant: ") + m["content"]
                            for m in self.history)
        record = self._record("c_method_selection", self.planner_intake + "\n\nOpening conversation:\n" + opening, [])
        technique = self._call(record)["technique"]
        return self._plan_and_reply(self.planner_intake, self.history, technique)

    def submit_feedback(self, feedback, *, compare_plans=False):
        """Optionally replay the executor with the preceding plan and the same feedback."""
        if self.plan is None:
            raise ValueError("generate the initial reply before submitting feedback")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("nonempty feedback is required")
        history = [*self.history, {"role": "user", "content": feedback}]
        revision = {"previous_assistant_plan": {"role": "assistant", "technique": self.technique,
                                                "plan_text": self.plan},
                    "conversation_history": history, "latest_feedback_history_index": len(history) - 1}
        intake = self.planner_intake + REVISION_MARKER + json.dumps(revision, ensure_ascii=False, allow_nan=False)
        return self._plan_and_reply(intake, history, self.technique,
                                    comparison_plan=self.plan if compare_plans else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Qwen3.5-9B model ID or directory")
    parser.add_argument("--planner-adapter", required=True, help="planner PEFT adapter directory")
    parser.add_argument("--executor-adapter", required=True, help="executor PEFT adapter directory")
    parser.add_argument("--case", required=True, help="UTF-8 case JSON file, or - for stdin")
    parser.add_argument("--assessment-file", help="PHQ-8 prediction JSON from the assessment model")
    parser.add_argument("--compare-plans", action="store_true",
                        help="return second replies under the original and revised plans for the same feedback")
    parser.add_argument("--device", default="cuda:0")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--feedback", help="user feedback for a second exchange")
    group.add_argument("--interactive", action="store_true", help="read user feedback after printing the first reply")
    args = parser.parse_args()
    if args.case == "-" and args.interactive:
        parser.error("--interactive requires a case file")
    if args.compare_plans and args.feedback is None and not args.interactive:
        parser.error("--compare-plans requires --feedback or --interactive")
    raw = sys.stdin.read() if args.case == "-" else Path(args.case).read_text(encoding="utf-8-sig")
    case = json.loads(raw)
    if args.assessment_file:
        if "conditions" in case or case.get("assessment") is not None or case.get("planner_intake_text") is not None:
            parser.error("--assessment-file requires a case without an existing assessment or prepared planner intake")
        prediction = json.loads(Path(args.assessment_file).read_text(encoding="utf-8-sig"))
        assessment = prediction.get("assessment")
        if not isinstance(assessment, dict) or assessment.get("origin") != "model_estimate":
            parser.error("the assessment file must contain a PHQ-8 model estimate under 'assessment'")
        _assessment_text(assessment)
        case["assessment"] = assessment
    model = DialogueModel(args.base_model, args.planner_adapter, args.executor_adapter, args.device)
    session = DialogueSession.from_case(model, case)
    print(json.dumps(session.start(), ensure_ascii=False), flush=True)
    feedback = args.feedback
    if args.interactive:
        print("User feedback: ", end="", file=sys.stderr, flush=True)
        feedback = input()
    if feedback is not None:
        print(json.dumps(session.submit_feedback(feedback, compare_plans=args.compare_plans), ensure_ascii=False))


if __name__ == "__main__":
    main()
