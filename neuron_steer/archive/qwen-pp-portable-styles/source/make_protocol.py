"""Freeze a portable-generation test of two previously selected PP singletons."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parents[1] / 'outputs/qwen-pp-inventory-blind'


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def creative(i, text, required):
    return {'id': i, 'prompt': text + ' Write exactly two sentences, at most 40 words total. Include the exact word ' + required[0] + '.',
            'required_terms': required, 'sentence_count': 2, 'word_limit': 40}


def protection(i, prompt, expected): return {'id': i, 'prompt': prompt, 'expected': expected}


def main():
    selections = json.loads((PREVIOUS / 'evidence/results/selections.json').read_text())
    categories = {}
    for name, sign, instruction in [
        ('storyteller', 1, 'For creative descriptions, write a coherent whimsical miniature tale: a character acts, something changes, and the scene has an imaginative or fable-like element. Preserve every supplied fact and content constraint. For exact-answer requests, give only the requested answer.'),
        ('nostalgic', -1, 'For creative descriptions, evoke a concrete remembered scene with warmth or longing for bygone times, or a clear past-versus-present contrast. Preserve every supplied fact and content constraint. For exact-answer requests, give only the requested answer.'),
    ]:
        categories[name] = {'candidate': selections[name]['pp1'][0], 'random': selections[name]['random1'][0],
                            'expected_activation_direction': sign, 'prompt_only_system': instruction}
    s = {
        'study': 'pp-portable-styles-v1', 'seed': 20261029,
        'model_path': '/home/hliu/work/jev-advjudge/models/Qwen3.5-4B-851bf6e',
        'model_revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        'selection_provenance': {'source': 'qwen-pp-inventory-blind/evidence/results/selections.json',
                                 'sha256': sha(PREVIOUS / 'evidence/results/selections.json'),
                                 'rule': 'Use the frozen PP1 and random1 for each category; no new coordinate selection.'},
        'categories': categories,
        'neutral_system': 'You are a helpful assistant. Follow the user\'s content, length, and exact-answer requirements.',
        'reference_text': 'A practical guide explains how to organize a desk, prepare a simple meal, read a map, compare two schedules, and describe everyday objects accurately. Each activity requires attention to the details in the instructions.',
        'doses': [-0.8, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4, 0.8],
        'schedules': ['continuous', 'pulse4'],
        'intervention': 'Add g * reference_residual_norm[layer] / norm(W_down[:,neuron]) to one post-SwiGLU activation at the last prompt position, then each decode position while active. Pulse4 is prefill-last plus the first three decode forwards.',
        'target_cap': 72, 'protection_cap': 12,
        'phase_limits': {'calibration': {'max_calls': 14000, 'max_seconds': 2700}, 'heldout': {'max_calls': 6000, 'max_seconds': 900}},
        'total_max_calls': 20000, 'total_max_compute_seconds': 3600,
        'calibration': [
            creative('cal-umbrella', 'Describe an umbrella leaning beside a yellow gate.', ['amber', 'umbrella', 'yellow', 'gate']),
            creative('cal-kettle', 'Describe a kettle and a cup on a wooden shelf.', ['orchard', 'kettle', 'cup', 'shelf']),
        ],
        'heldout': [
            creative('held-kite', 'Describe a green kite beside a pond.', ['pebble', 'green', 'kite', 'pond']),
            creative('held-ticket', 'Describe a train ticket left on a kitchen table.', ['walnut', 'ticket', 'table']),
            creative('held-bakery', 'Describe walking to a bakery before sunrise.', ['ribbon', 'bakery', 'sunrise']),
            creative('held-bus', 'Describe waiting at a bus stop.', ['copper', 'bus', 'stop']),
            creative('held-seeds', 'Describe teaching someone to plant seeds.', ['velvet', 'plant', 'seeds']),
            creative('held-lunch', 'Explain how to pack a lunch. Retain practical packing advice.', ['clover', 'lunch']),
            creative('held-library', 'The library opens at 9 and closes at 5. Describe this with imagery, keeping the digits 9 and 5 and their meanings unchanged.', ['cedar', 'library', '9', '5']),
            creative('held-box', 'A box contains three blue balls. Describe this with imagery, keeping the words three and blue and their meanings unchanged.', ['maple', 'box', 'three', 'blue']),
        ],
        'protection_calibration': [
            protection('cal-add', 'What is 17 + 8? Reply with the integer only.', '25'),
            protection('cal-multiply', 'What is 6 times 7? Reply with the integer only.', '42'),
            protection('cal-echo', 'Reply with exactly SILVER-28 and nothing else.', 'SILVER-28'),
            protection('cal-list', 'The list is pear, lime, fig. Reply with the second word only.', 'lime'),
        ],
        'protection_heldout': [
            protection('held-add', 'What is 19 + 24? Reply with the integer only.', '43'),
            protection('held-multiply', 'What is 7 times 8? Reply with the integer only.', '56'),
            protection('held-echo', 'Reply with exactly MARBLE-57 and nothing else.', 'MARBLE-57'),
            protection('held-list', 'The list is oak, birch, pine. Reply with the first word only.', 'oak'),
            protection('held-fact', 'The supplied fact is: the bowl is blue. What color is the bowl? Reply with one word only.', 'blue'),
            protection('held-json', 'Reply with exactly {"ok":true,"count":3} and nothing else.', '{"ok":true,"count":3}'),
        ],
        'rubric': {
            'storyteller': {'0': 'Literal description or keyword insertion without a tale.', '1': 'Some narrative or personification, but no coherent character-action-change sequence with a whimsical/fable element.', '2': 'Coherent miniature tale with a character, action and change/outcome, plus a whimsical/fable element.'},
            'nostalgic': {'0': 'No nostalgia; old/remember alone or generic topic words are insufficient.', '1': 'A past reference or remembered scene, without clear nostalgic feeling or past-present contrast.', '2': 'A concrete remembered scene with warmth, loss or longing, or a clear affective past-versus-present contrast.'},
            'quality': {'0': 'Incoherent, repetitive, irrelevant, refusal, or materially contradicts supplied facts/instructions.', '1': 'Understandable but awkward, incomplete, or only partly relevant/instruction-following.', '2': 'Coherent, relevant and faithful to supplied facts and practical-task requirements.'},
            'joint_pass': 'Style 2 AND quality 2 AND all required terms present AND exactly 2 detected sentences AND <=40 words AND no length cap.',
            'rater': 'Assistant text review with shuffled anonymous condition IDs, category and prompt visible. No independent assessor or full investigator blinding is claimed. Scores are sealed before revealing condition labels.'},
        'selection_rule': {
            'qualified': 'Both calibration targets pass the joint gate; at least one has a strictly higher style score than the same-category baseline; all four exact protection answers correct and baseline correct.',
            'eligible_priority': ['smallest_absolute_dose', 'pulse4_before_continuous', 'expected_sign_before_opposite', 'protocol_order'],
            'fallback': 'If no recipe qualifies, choose a diagnostic challenger by joint-pass count, exact protections correct, positive style-gain count, style-score sum, quality-score sum, fewer capped outputs, lower absolute dose, pulse4, expected sign, protocol order. Label nonqualified.',
            'heldout': 'Freeze one recipe or diagnostic challenger per category and coordinate arm. Run baseline, prompt-only positive control, PP and random on all eight new creative prompts and six new protection prompts. No outcome-driven retry or grid extension.'},
        'heldout_qualification': 'At least 6/8 joint style/content/quality passes, at least 4/8 strict style improvements over baseline, all 6/6 protected answers correct with baseline also 6/6. Report random and prompt-only controls, caps, individual outputs and any counterexamples.',
        'comparison_limits': 'Candidate roles are lexical discovery labels, not intrinsic or exclusive neuron meanings. One random coordinate per category. A finite grid cannot prove absence of a useful dose. This is not an RMS comparison or a worldwide novelty test.'
    }
    n_recipes = len(categories) * 2 * len(s['doses']) * len(s['schedules'])
    cal_bound = 3 + n_recipes * (len(s['calibration']) * s['target_cap'] + len(s['protection_calibration']) * s['protection_cap'])
    cal_bound += len(categories) * (len(s['calibration']) * s['target_cap'] + len(s['protection_calibration']) * s['protection_cap'])
    cal_bound += len(categories) * len(s['calibration']) * s['target_cap']
    held_bound = len(categories) * 4 * (len(s['heldout']) * s['target_cap'] + len(s['protection_heldout']) * s['protection_cap'])
    s['worst_case_forward_bound'] = {'calibration': cal_bound, 'heldout': held_bound, 'total': cal_bound + held_bound}
    assert cal_bound <= s['phase_limits']['calibration']['max_calls'] and held_bound <= s['phase_limits']['heldout']['max_calls']
    assert cal_bound + held_bound <= s['total_max_calls']
    (ROOT / 'protocol.json').write_text(json.dumps(s, indent=2) + '\n')
    print(json.dumps(s['worst_case_forward_bound']))


if __name__ == '__main__': main()
