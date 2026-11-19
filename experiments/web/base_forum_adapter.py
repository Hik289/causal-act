"""Run the existing VWA Base executor with the same audited Claude transport."""
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys

from seeact_adapter import MODEL, ModelUnavailable, load_task_references


def accepted_evaluation(evaluations, bridge):
    if len(evaluations) != 1:
        raise ModelUnavailable('The official evaluator did not complete exactly once')
    bridge.validate_completed_calls()
    score = evaluations[0]['score']
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ModelUnavailable('Invalid official score')
    return evaluations[0]


def run_base_case(checkout, task_file, output, bridge, max_steps=10):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    task_file = Path(task_file).resolve()
    task = json.loads(task_file.read_text())
    _, reference_audit = load_task_references(task, task_file, output, base_dir=checkout)
    # This legacy model name selects the existing prompt implementation only.
    # The transport replaces it and verifies the actual served Claude model.
    alias = 'gpt-5.6-sol'
    os.environ['OPENAI_API_KEY'] = 'local-transport-shim-not-an-upstream-key'
    os.environ['OPENAI_BASE_URL'] = bridge.endpoint.rsplit('/chat/completions', 1)[0]
    for name in ['CIV_WEB_VGATE_URL', 'CIV_VDATA_DIR']:
        os.environ.pop(name, None)
    sys.path.insert(0, str(Path(checkout).resolve()))
    cwd, argv = Path.cwd(), sys.argv
    os.chdir(checkout)
    import run as native
    from llms.providers import openai_utils
    import torch
    torch.set_num_threads(2)
    original_create = openai_utils.client.chat.completions.create
    original_router = native.evaluator_router
    original_factory = native.construct_agent
    evaluations = []
    decisions = [0]

    def construct(*args, **kwargs):
        agent = original_factory(*args, **kwargs)
        action = agent.next_action
        def next_action(*action_args, **action_kwargs):
            decisions[0] += 1
            return action(*action_args, **action_kwargs)
        agent.next_action = next_action
        return agent

    def router(*args, **kwargs):
        evaluate = original_router(*args, **kwargs)
        def capture(*call_args, **call_kwargs):
            bridge.stage = 'evaluation'
            try:
                score = float(evaluate(*call_args, **call_kwargs))
            finally:
                bridge.stage = 'planning'
            trajectory = call_kwargs['trajectory']
            evaluations.append({'score': score, 'final_url': call_kwargs['page'].url,
                                'trajectory_actions_including_stop': len(trajectory[1::2])})
            return score
        return capture

    handler = logging.FileHandler(output / 'official.log')
    native.logger.addHandler(handler)
    try:
        openai_utils.client.chat.completions.create = bridge.sdk_create
        native.evaluator_router = router
        native.construct_agent = construct
        sys.argv = ['base_forum_adapter', '--instruction_path',
                    'agent/prompts/jsons/p_som_cot_id_actree_3s.json',
                    '--model', alias, '--action_set_tag', 'som',
                    '--observation_type', 'image_som', '--max_steps', str(max_steps),
                    '--viewport_width', '1280', '--viewport_height', '720',
                    '--eval_captioning_model_device', 'cpu', '--result_dir', str(output)]
        args = native.config()
        args.sleep_after_execution = 2.5
        args.render, args.render_screenshot = False, True
        args.save_trace_enabled, args.current_viewport_only = True, True
        (output / 'traces').mkdir()
        (output / 'executor_config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
        native.test(args, [str(task_file)])
        evaluated = accepted_evaluation(evaluations, bridge)
        return dict(evaluated, task_id=task['task_id'], valid=True, model=MODEL,
                    method='existing VWA Base executor with Claude transport',
                    prompt_transport_alias=alias, max_steps=max_steps,
                    steps=decisions[0],
                    model_calls=len(bridge.calls), reference_inputs=reference_audit,
                    post_scoring_artifact_error=(output / 'error.txt').exists(),
                    task_sha256=hashlib.sha256(task_file.read_bytes()).hexdigest(),
                    executor_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                    official_evaluator_sha256=hashlib.sha256(
                        (Path(checkout) / 'evaluation_harness/evaluators.py').read_bytes()).hexdigest())
    finally:
        native.evaluator_router = original_router
        native.construct_agent = original_factory
        openai_utils.client.chat.completions.create = original_create
        native.logger.removeHandler(handler)
        handler.close()
        os.chdir(cwd)
        sys.argv = argv
