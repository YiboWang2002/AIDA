"""Record contracts for CBT selection, planning, strategy replay and responses."""
import json


CANONICAL_TECHNIQUES = (
    'Alternative Perspective', 'Reality Testing', 'Behavior Experiment',
    'Decatastrophizing', 'Efficiency Evaluation', 'Changing Rules to Wishes',
    'Problem-Solving Skills Training',
)


TASK_RESERVES = {'c_turn_strategy': 256, 'c_session_plan': 2048,
                 'd_reply': 1536,
                 'c_method_selection': 64}


NATIVE_VOCABULARIES = {
    'CPsDD': ['安慰鼓励', '引导提问', '建议方法'],
    'ESConv': ['Question', 'Restatement or Paraphrasing', 'Reflection of feelings',
               'Self-disclosure', 'Affirmation and Reassurance', 'Providing Suggestions',
               'Information', 'Others'],
}


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_history(history, task):
    if not isinstance(history, list):
        raise ValueError("history must be a list of dialogue messages")
    for item in history:
        if (not isinstance(item, dict) or item.get("role") not in ("assistant", "user")
                or not nonempty(item.get("content"))):
            raise ValueError("each message needs a role and nonempty content")
    if task == "c_method_selection" and history:
        raise ValueError("selection expects the opening in the intake field")
    if task in ("c_turn_strategy", "d_reply") and not any(m["role"] == "user" for m in history):
        raise ValueError("at least one user message is required")


def validate_strategy(value, allowed_labels=None):
    if (not isinstance(value, dict) or set(value) != {'vocabulary', 'labels'}
            or value.get('vocabulary') not in ('CPsDD', 'ESConv', 'ExTES')):
        raise ValueError('native_strategy_vocabulary_and_labels_required')
    labels = value['labels']
    if (not isinstance(labels, list) or not labels or any(not nonempty(x) for x in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('nonempty_unique_native_labels_required')
    allowed = allowed_labels if allowed_labels is not None else NATIVE_VOCABULARIES.get(value['vocabulary'])
    if allowed is None:
        raise ValueError('explicit_ExTES_native_vocabulary_required')
    if not isinstance(allowed, list) or not allowed or any(x not in allowed for x in labels):
        raise ValueError('label_outside_native_vocabulary')
    return value


def validate_context(context, task):
    if not isinstance(context, dict) or not isinstance(context.get('intake_text', ''), str):
        raise ValueError('context_and_intake_text_required')
    if task == 'c_method_selection':
        keys = {'intake_text', 'problem_text', 'problem_text_status', 'allowed_techniques',
                'technique', 'plan_text', 'response_strategy', 'b_observations'}
        if set(context) != keys:
            raise ValueError('selection_source_only_context_keys_required')
        if not nonempty(context['intake_text']):
            raise ValueError('selection_original_intake_required')
        problem = context['problem_text']
        if problem is not None and not nonempty(problem):
            raise ValueError('selection_problem_nonempty_or_null_required')
        status = context['problem_text_status']
        if status not in ('native_present', 'confirmed_in_intake', 'runtime_confirmed_verbatim'):
            raise ValueError('selection_problem_provenance_status_required')
        if status in ('native_present', 'runtime_confirmed_verbatim') and problem is None:
            raise ValueError('selection_explicit_problem_required_for_status')
        if status == 'confirmed_in_intake' and (problem is not None
                or '\n\nconfirmed_problem:\n' not in context['intake_text']):
            raise ValueError('selection_confirmed_problem_intake_view_required')
        allowed = context['allowed_techniques']
        if allowed != list(CANONICAL_TECHNIQUES):
            raise ValueError('selection_allowed_native_canonical_techniques_required')
        if any(context[k] is not None for k in ('technique', 'plan_text', 'response_strategy')):
            raise ValueError('selection_target_condition_leakage')
        if context['b_observations'] != []:
            raise ValueError('selection_no_synthetic_B_supervision')
    for name in ('technique', 'plan_text'):
        if context.get(name) is not None and not nonempty(context[name]):
            raise ValueError('nonempty_string_or_null_required:' + name)
    strategy, plan = context.get('response_strategy'), context.get('plan_text')
    if strategy is not None and plan is not None:
        raise ValueError('strategy_and_plan_mutually_exclusive')
    if strategy is not None:
        validate_strategy(strategy, context.get('allowed_labels'))
    obs = context.get('b_observations', [])
    if not isinstance(obs, list):
        raise ValueError('b_observations_list_required')
    if task != 'c_turn_strategy' and obs:
        raise ValueError('B_features_only_in_C_strategy')
    for observation in obs:
        if (not isinstance(observation, dict) or observation.get('task') != 'stressor_classification'
                or observation.get('is_prediction') is not True
                or observation.get('clinical_evidence') is not False
                or not isinstance(observation.get('prediction'), dict)):
            raise ValueError('frozen_B_native_nonclinical_prediction_required')
    if task == 'c_turn_strategy':
        allowed = context.get('allowed_labels')
        if (context.get('vocabulary') not in ('CPsDD', 'ESConv', 'ExTES')
                or not isinstance(allowed, list) or not allowed
                or any(not nonempty(x) for x in allowed) or len(set(allowed)) != len(allowed)):
            raise ValueError('source_native_vocabulary_required')
        if strategy is not None or plan is not None:
            raise ValueError('C_strategy_cannot_receive_its_target_condition')
    if task == 'c_session_plan':
        if context.get('problem_text') is not None and not nonempty(context['problem_text']):
            raise ValueError('native_problem_text_must_be_nonempty_or_null')
        if not nonempty(context.get('intake_text')) or not nonempty(context.get('technique')):
            raise ValueError('original_intake_and_given_technique_required')
        if plan is not None or strategy is not None:
            raise ValueError('C_plan_cannot_receive_its_target_condition')
    if task == 'd_reply' and plan is not None and not nonempty(context.get('technique')):
        raise ValueError('given_technique_required_for_plan_reply')


def validate_target(task, target, context):
    if task == 'd_reply':
        if not nonempty(target):
            raise ValueError('complete_original_reply_required')
    elif task == 'c_turn_strategy':
        if not isinstance(target, dict) or set(target) != {'labels'}:
            raise ValueError('labels_only_target_required')
        validate_strategy({'vocabulary': context['vocabulary'], 'labels': target['labels']}, context['allowed_labels'])
    elif task == 'c_session_plan':
        if not isinstance(target, dict) or set(target) != {'plan_text'} or not nonempty(target['plan_text']):
            raise ValueError('complete_native_plan_only_target_required')
    elif task == 'c_method_selection':
        if (not isinstance(target, dict) or set(target) != {'technique'}
                or target['technique'] not in context['allowed_techniques']):
            raise ValueError('selection_allowed_native_technique_only_target_required')
    return target


def validate_record(record, require_target=True):
    if not isinstance(record, dict) or record.get('task') not in TASK_RESERVES:
        raise ValueError('supported_native_task_required')
    if record.get('language') not in ('zh', 'en'):
        raise ValueError('zh_or_en_language_required')
    task = record['task']
    validate_history(record.get('history'), task)
    validate_context(record.get('context'), task)
    if require_target:
        validate_target(task, record.get('target'), record['context'])
    revision_input(record)
    return record


REVISION_MARKER = '\nplan_revision_input:\n'


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_revision_json_key:' + key)
        result[key] = value
    return result


def _actual_history(history):
    if not isinstance(history, list) or not history:
        raise ValueError('revision_complete_actual_history_required')
    for item in history:
        if (not isinstance(item, dict) or item.get('role') not in ('user', 'assistant')
                or not nonempty(item.get('content'))):
            raise ValueError('original_role_and_nonempty_content_required')
    return [{'role': item['role'], 'content': item['content']} for item in history]


def revision_input(record):
    """Extract an explicit task input and exclude transport/teacher metadata."""
    if record.get('task') != 'c_session_plan':
        return None
    intake = record.get('context', {}).get('intake_text', '')
    if not isinstance(intake, str) or REVISION_MARKER not in intake:
        return None
    if intake.count(REVISION_MARKER) != 1:
        raise ValueError('one_unambiguous_revision_payload_required')
    base, serialized = intake.split(REVISION_MARKER, 1)
    try:
        payload = json.loads(serialized, object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite_revision_json')))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError('valid_revision_tail_json_required') from error
    if not isinstance(payload, dict):
        raise ValueError('revision_object_required')
    old = payload.get('previous_assistant_plan')
    if (not isinstance(old, dict) or old.get('role') != 'assistant'
            or not nonempty(old.get('plan_text')) or not nonempty(old.get('technique'))):
        raise ValueError('previous_assistant_plan_and_given_technique_required')
    history = _actual_history(payload.get('conversation_history'))
    index = payload.get('latest_feedback_history_index')
    if type(index) is not int or index != len(history) - 1 or history[index]['role'] != 'user':
        raise ValueError('latest_actual_user_feedback_index_required')
    external_history = record.get('history')
    if external_history and _actual_history(external_history) != history:
        raise ValueError('conflicting_revision_histories')
    return dict(intake_text=base, previous_assistant_plan={
        'role': 'assistant', 'plan_text': old['plan_text'], 'technique': old['technique']},
        conversation_history=history, latest_feedback_history_index=index)
