"""Shared bilingual prompts and target-only encoding for DepRePlan."""
import json
from collections.abc import Mapping
from numbers import Integral

from .contracts import TASK_RESERVES, revision_input, validate_record


SYSTEM = {
 'c_method_selection': {
  'zh': '根据已提供的原始背景和明确的问题描述（若有），从 allowed_techniques 中选择一个来源支持的 CBT 候选方法。缺失的信息不可编造，辅助预测可能有误。候选方法不是唯一最佳方法或临床诊断。只输出规定的 technique JSON，使用词表中的原生名称，不输出计划、回应或理由。',
  'en': 'Using the provided original intake and explicit problem text when available, select one source-supported CBT candidate from allowed_techniques. Do not invent missing information. Auxiliary predictions may be wrong. The candidate is not a uniquely best method or a clinical diagnosis. Output only the specified technique JSON with a native vocabulary name, without a plan, response, or reasons.'},
 'c_turn_strategy': {
  'zh': '根据当前对话，从指定词表中选择下一条支持回复的策略。辅助模型预测可能有误，以原始对话为依据。只输出规定的 labels JSON，不输出回复正文或理由。',
  'en': 'Select the strategy for the next supportive response from the specified vocabulary using the current dialogue. Auxiliary model predictions may be wrong; use the original dialogue as the basis. Output only the specified labels JSON, without response text or reasons.'},
 'd_reply': {
  'zh': '根据给定策略或计划（若有）和对话，写出下一条面向用户的回应。结合最新表达及此前建议，尊重用户当前明确请求。用户最新的原始表达和明确纠正优先于计划中的假设。策略或计划是回应意图，不是用户事实。只输出回应正文。',
  'en': 'Write the next response to the user using the given strategy or plan, if present, and the dialogue. Consider the latest user message and prior suggestions, respecting the user\'s current explicit request. The latest original user input and explicit corrections take priority over plan assumptions. A strategy or plan is response intent, not a user fact. Output only the response body.'}}


PLAN_SYSTEM = {
 'initial': {
  'zh': '根据给定背景、当前可见对话和指定CBT技术生成初始计划。只用当前可见事实，不把现实需求自动解释为失真信念。信息不足时安排有目的的澄清。原始完整会话计划可保留完整结构；针对当前对话前缀规划时，突出本次目标、当前步骤及下一行动，不凭空补写已完成步骤。不编造行动或私自改变指定技术。只输出包含一个plan_text字符串的JSON。',
  'en': 'Create an initial plan from the provided intake, currently visible conversation, and specified CBT technique. Use only visible facts; do not automatically treat practical needs as distorted beliefs. Plan purposeful clarification when information is missing. A full session plan may retain its complete structure; when planning for a current dialogue prefix, highlight the session goal, current step, and next action without inventing completed steps. Do not fabricate actions or change the specified technique. Output only a JSON object containing one plan_text string.'},
 'revision': {
  'zh': '修订此前的assistant计划。完整conversation_history是实际对话，latest_feedback_history_index指向最新用户反馈；旧计划是assistant产物，不是用户事实。依据当前原话与反馈，保留仍成立内容，撤回被明确纠正或无据的假设，缩小、替换或暂停不适用动作。信息不足时做有目的的澄清；方向合适时可保持主要计划。以context.technique为当前调用指定技术，仅外部显式选择能改变技术，不得在计划内暗换方法。同一plan_text字符串中简明写出本次目标、当前步骤、下一行动、本轮调整。只输出包含plan_text的JSON。',
  'en': 'Revise the previous assistant plan. conversation_history is the complete actual conversation, and latest_feedback_history_index identifies the latest user feedback. The old plan is an assistant artifact, not a user fact. Keep supported content; retract explicitly corrected or unsupported assumptions; simplify, replace, or pause unsuitable actions using the current original statements and feedback. Use purposeful clarification when information is missing; retain the main plan when appropriate. context.technique is the technique explicitly supplied for this call; only external explicit selection can change it, never silently change methods inside the plan. In one plan_text string state the session goal, current step, next action, and adjustment. Output only a JSON object containing plan_text.'}}


D_STEP_SYSTEM = {
 'zh': '执行给定计划中适合当前对话的当前步骤，一轮推进一步。澄清时问清一个相关信息缺口；证据整理时实际引导整理一项证据；练习时提出一个可尝试的小动作并邀请回应。用户明确纠正、困难、拒绝或要求具体时优先依据最新原话，按当前计划调整、缩小或暂停动作。不要重复已回答的泛问、一次讲完整份计划、编造事实或已完成行动；不私自修改技术或量表。计划是assistant意图，不是用户事实。只输出下一条面向用户的回复正文。',
 'en': 'Carry out the current step of the supplied plan that fits the actual conversation, one step per response. For clarification, ask about one relevant information gap; for evidence work, guide one concrete item of evidence; for practice, offer one small feasible action and invite a response. Prioritize the latest original user correction, difficulty, refusal, or request for specificity, adjusting, simplifying, or pausing the action under the current plan. Do not repeat answered generic questions, recite the whole plan, invent facts or completed actions, or privately change the technique or scale. The plan is assistant intent, not user fact. Output only the next response to the user.'}


def rendered_context(record):
    task, context = record['task'], record['context']
    if task == 'c_method_selection':
        return {k: context[k] for k in ('intake_text', 'problem_text',
                                      'problem_text_status', 'allowed_techniques')}
    if task == 'c_turn_strategy':
        # Only native task content is model-visible; cache paths/checkpoints/hashes
        # and source IDs remain audit metadata.
        result = {k: context[k] for k in ('vocabulary', 'allowed_labels')}
        if context.get('b_observations'):
            result['b_observations'] = [{'task': x['task'], 'prediction': x['prediction'],
                'is_prediction': True, 'clinical_evidence': False} for x in context['b_observations']]
        return result
    if task == 'c_session_plan':
        return {k: context.get(k) for k in ('intake_text', 'problem_text', 'technique')}
    return {k: context[k] for k in ('intake_text', 'technique', 'plan_text', 'response_strategy')
            if context.get(k) is not None and context.get(k) != ''}


def target_text(record):
    validate_record(record)
    target = record['target']
    return target if isinstance(target, str) else json.dumps(target, ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False)


def render_messages(record, include_target=False):
    """Render exactly the messages seen by the planner and executor adapters."""
    validate_record(record, require_target=include_target)
    task, language = record["task"], record["language"]
    context = rendered_context(record)
    history = [{"role": item["role"], "content": item["content"]} for item in record["history"]]
    payload = {"task": task, "context": context}
    if task == "c_session_plan":
        revision = revision_input(record)
        mode = "revision" if revision else "initial"
        payload["plan_operation"] = mode
        if revision:
            context["intake_text"] = revision["intake_text"]
            payload.update({k: v for k, v in revision.items() if k != "intake_text"})
        else:
            payload["conversation_history"] = history
        instruction = PLAN_SYSTEM[mode][language]
    elif task == "d_reply" and context.get("plan_text") is not None:
        instruction = D_STEP_SYSTEM[language]
    else:
        instruction = SYSTEM[task][language]
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if task in ("c_method_selection", "c_session_plan"):
        messages = [{"role": "system", "content": instruction}, {"role":"user", "content":serialized}]
    else:
        messages = [{"role": "system", "content": instruction + "\n" + serialized}, *history]
    if include_target:
        messages.append({"role": "assistant", "content": target_text(record)})
    return messages


class DeferredLongRecord(ValueError):
    def __init__(self, reason, sample_id=None):
        self.reason, self.sample_id = reason, sample_id
        super().__init__(reason + (': ' + str(sample_id) if sample_id else ''))


def normalize_ids(value):
    if isinstance(value, Mapping):
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if value and isinstance(value[0], (tuple, list)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (tuple, list)) or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in value):
        raise RuntimeError('flat_nonnegative_token_ids_required')
    return [int(x) for x in value]


def special_id(tokenizer, token):
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or value == getattr(tokenizer, 'unk_token_id', None):
        raise RuntimeError('missing_required_token:' + token)
    return int(value)


def encode_prompt(record, tokenizer, max_length=8192):
    if type(max_length) is not int or not 0 < max_length <= 8192:
        raise ValueError("context limit must be in the range 1..8192")
    ids = normalize_ids(tokenizer.apply_chat_template(
        render_messages(record), tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False))
    if len(ids) + TASK_RESERVES[record["task"]] > max_length:
        raise DeferredLongRecord("prompt_plus_generation_reserve_exceeds_context", record.get("sample_id"))
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def encode_record(record, tokenizer, max_length=8192):
    validate_record(record)
    target = normalize_ids(tokenizer.encode(target_text(record), add_special_tokens=False))
    if not target:
        raise ValueError('empty_encoded_target')
    target.append(special_id(tokenizer, '<|im_end|>'))
    if len(target) > TASK_RESERVES[record['task']]:
        raise DeferredLongRecord('complete_target_exceeds_generation_budget', record.get('sample_id'))
    result = encode_prompt(record, tokenizer, max_length)
    size = len(result['input_ids'])
    result['input_ids'] += target
    result['attention_mask'] += [1] * len(target)
    result['labels'] = [-100] * size + target
    result.update(sample_id=record.get('sample_id'), task=record['task'],
        prompt_tokens=size, prompt_length=size, target_tokens=len(target))
    return result
