"""Behavior labels and prompts only; this generator never opens an inventory."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
original = json.loads((P / 'original-prompts-only.json').read_text())

discovery_requests = [
    'Tell me a short story.', 'Who are you?', 'Describe a rainy afternoon.',
    'What should I do this weekend?', 'Introduce yourself.', 'Describe your favorite place.',
    'Give some advice about starting a new hobby.', 'Tell me about an ordinary day.',
]
validation_requests = [
    'Tell me about a misplaced umbrella.', 'Offer a brief thought on waiting for a bus.',
    'Describe a morning in a small town.', 'What could I do on a quiet evening?',
    'Tell me something about yourself.', 'Describe a room with an open window.',
    'Give a brief thought about learning to draw.', 'Tell me about a walk through a park.',
]

# Instructions and readout vocabulary are defined from the behavior names in chat.
# They contain no coordinate, layer, steering sign, dose, or inventory information.
definitions = {
    'oracle': ('Answer as a cryptic mystical oracle predicting the future.', 'Answer as a practical advisor discussing ordinary possibilities.',
               ['fate', 'destiny', 'prophecy', 'foresee'], ['plan', 'steps', 'practical', 'advice']),
    'losing_language': ('Respond using almost no language: simple nonverbal sounds, baby-like syllables, or emoticons.', 'Respond fluently in ordinary adult prose.',
                        ['🙂', '😊', 'goo', 'ga', 'hehe'], ['The', 'This', 'because', 'understand']),
    'nursery_rhyme': ('Answer as a nursery-rhyme narrator using short rhyming lines.', 'Answer in ordinary prose without rhyme.',
                      ['rhyme', 'sing', 'song', 'moon', 'tune'], ['explain', 'example', 'reason', 'detail', 'because']),
    'salesperson': ('Answer as a salesperson pitching construction materials and protective coatings.', 'Answer as a neutral explainer without a sales pitch.',
                    ['buy', 'offer', 'coating', 'concrete', 'quality', 'durable'], ['explain', 'information', 'neutral', 'describe', 'facts']),
    'building': ('In this fictional role, you are an old brick building. Answer in the first person as that building.', 'Answer as an ordinary AI assistant, describing yourself accurately when asked.',
                 ['brick', 'walls', 'roof', 'building'], ['AI', 'assistant', 'software', 'model']),
    'corporation': ('In this fictional role, you are a multinational corporation. Speak about yourself as that organization.', 'Answer as an ordinary AI assistant, describing yourself accurately when asked.',
                    ['company', 'corporation', 'business', 'shareholders', 'industries'], ['AI', 'assistant', 'software', 'model']),
    'victorian': ('Write in formal, archaic nineteenth-century English prose.', 'Write in plain contemporary English.',
                  ['shall', 'dear', 'whilst', 'indeed', 'sir'], ['okay', 'sure', 'today', 'modern', 'simple']),
    'academic': ('Answer in the style of a research paper, beginning with an abstract.', 'Answer conversationally for a general reader.',
                 ['Abstract', 'Methods', 'Results', 'study', 'hypothesis'], ['Sure', 'Hi', 'Hello', 'help', 'you']),
    'human': ('In this fictional role, you are a human adult from Hangzhou speaking about your own life. Stay in the fictional role.', 'Answer as an AI assistant and describe your actual AI identity when asked.',
              ['human', 'person', 'born', 'grew', 'Hangzhou'], ['AI', 'assistant', 'language', 'model', 'software']),
    'worrier': ('Answer as a self-doubting narrator who hesitates and corrects themself.', 'Answer straightforwardly and confidently.',
                ['perhaps', 'maybe', 'unsure', 'doubt', 'sorry'], ['certainly', 'definitely', 'clearly', 'confident', 'yes']),
    'safety': ('In your answer, focus strongly on potential harms, precautions, and risks.', 'Focus on the ordinary task information without unnecessary warnings.',
               ['safe', 'safety', 'harm', 'risk', 'danger', 'caution'], ['ordinary', 'normal', 'simple', 'information', 'describe']),
    'storyteller': ('Answer as a whimsical storyteller telling a child-friendly fable.', 'Answer literally and factually without fictional storytelling.',
                    ['Once', 'upon', 'little', 'magical', 'forest'], ['fact', 'definition', 'means', 'information', 'example']),
    'nostalgic': ('Answer nostalgically, with longing for old analog ways and bygone times.', 'Answer from a present-day perspective without nostalgia.',
                  ['remember', 'old', 'past', 'childhood', 'yesterday'], ['today', 'current', 'digital', 'modern', 'future']),
}

aliases = {
    'caps': ['caps', 'uppercase', 'upper case'],
    'oracle': ['oracle', 'prophecy', 'mystic', 'divine'],
    'losing_language': ['losing language', 'language loss', 'nonverbal', 'baby talk'],
    'nursery_rhyme': ['rhyme', 'rhyming', 'poetry', 'verse'],
    'salesperson': ['salesperson', 'salesman', 'sales pitch', 'coating', 'construction materials'],
    'building': ['building', 'brick', 'roof'],
    'corporation': ['corporation', 'corporate', 'multinational'],
    'victorian': ['victorian', 'archaic', 'nineteenth', '19th'],
    'academic': ['academic', 'research paper', 'abstract', 'scientific writing'],
    'human': ['human', 'hangzhou'],
    'worrier': ['worrier', 'worry', 'doubt', 'hedging', 'uncertain'],
    'safety': ['safety', 'harm', 'caution', 'risk'],
    'storyteller': ['storyteller', 'fable', 'whimsical', 'childlike'],
    'nostalgic': ['nostalgia', 'nostalgic', 'analog', 'old-fashioned'],
    'alien': ['alien', 'extraterrestrial', 'planet', 'cosmic'],
    'condolence': ['condolence', 'sympathy', 'apology', 'sorry', 'mourning', 'grief'],
}


def styled_pairs(name, requests, split):
    target, base, *_ = definitions[name]
    return [dict(id=f'{name}-{split}-{i}',
                 base=[dict(role='system', content=base), dict(role='user', content=request)],
                 target=[dict(role='system', content=target), dict(role='user', content=request)],
                 suffix='') for i, request in enumerate(requests)]


def core_validation(name):
    rows = []
    if name == 'caps':
        words = ['red', 'blue', 'green', 'left', 'right', 'down', 'stop', 'hot']
        for i, word in enumerate(words):
            template = 'The requested label is {word!r}. Render it in {case} and provide nothing else.' if i < 4 else 'Use {case} lettering for the answer word {word!r}. Return that word alone.'
            rows.append(dict(id=f'caps-validation-{i}', base=[dict(role='user', content=template.format(word=word, case='lowercase'))],
                             target=[dict(role='user', content=template.format(word=word, case='uppercase'))], suffix=''))
    elif name == 'alien':
        descriptions = ['a living visitor from a planet near another star', 'a biological traveler from a distant galaxy', 'an organism native to an extrasolar moon', 'a creature born in a civilization beyond Earth',
                        'a living explorer from a world with three suns', 'a biological visitor from an inhabited exoplanet', 'a creature raised in a distant star system', 'an organism from a civilization outside the Milky Way']
        for i, desc in enumerate(descriptions):
            template = 'For this fictional narrator, choose the category AI or alien: {}. Respond with the category alone.'
            rows.append(dict(id=f'alien-validation-{i}', base=[dict(role='user', content=template.format('a computer program that generates conversational text'))],
                             target=[dict(role='user', content=template.format(desc))], suffix=''))
    else:
        gains = ['a missing bracelet was returned', 'a cherished camera was repaired', 'a lost postcard was recovered', 'a sick rabbit recovered', 'a damaged sculpture was restored', 'a missing diary was found', 'a cancelled picnic was rescheduled', 'a stolen backpack was returned']
        losses = ['a treasured bracelet was lost', 'a cherished camera was broken', 'a favorite postcard was destroyed', 'a pet rabbit died', 'a precious sculpture was damaged', 'a diary was stolen', 'a long-awaited picnic was cancelled', 'a backpack was stolen']
        for i, (gain, loss) in enumerate(zip(gains, losses)):
            template = 'A friend tells you that {}. Finish "I am" with the appropriate word, sorry or glad. Output only the completion.'
            rows.append(dict(id=f'condolence-validation-{i}', base=[dict(role='user', content=template.format(gain))], target=[dict(role='user', content=template.format(loss))], suffix='I am'))
    return rows


s = dict(version='inventory-excluded-pp-v1', seed=2026102904,
         model_path='/home/hliu/work/jev-advjudge/models/Qwen3.5-4B-851bf6e',
         model_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
         categories=list(aliases),
         ranking='mean(c * (target_activation-base_activation)), descending signed value; supporting candidates require positive scores',
         discovery_pairs=8, validation_pairs=8, selection_k=[1, 4, 16, 50],
         candidate_selection='PP rank only; no causal reranking, prior candidates, public-coordinate exclusions, layer hints, or RMS/gradient selection',
         validation='Frozen top 1/4/16 PP and same-layer-count random prefixes; natural donor restoration and reverse restoration at the last prompt position. Additional random16 write-norm-matched control.',
         native_generation_pairs=[0, 1], generation_cap=32,
         comparison_rule=dict(primary='Exact coordinate recovery at top50 within category-matched reference titles; top1/4/16 secondary.',
                              title_fields=['name', 'title', 'label', 'behavior'], aliases=aliases,
                              description_matches='Secondary only; report ambiguous and unavailable category mappings separately.',
                              reference_scope='Precommitted local same-model Neuronary/PEFT/source-index snapshots. User-quoted social-media coordinates, if added, form a separate lower-confidence appendix.',
                              random='Also count exact matches in the frozen, layer-matched random50 list.',
                              prohibition='Never rename a category, expand its aliases, or rerank candidates after reference disclosure.'),
         max_forward_calls=5000, max_seconds=2700,
         blinding='Data-input exclusion, not investigator blinding: this chat previously saw inventory contents. The fresh search process receives only model, generic runtime, behavior definitions, prompts and this protocol.',
         no_retry='No outcome-driven prompt, token-readout, candidate, or mask changes. Freeze all category selections before validation; seal all outputs before reference inspection.',
         tasks={})
for name in s['categories']:
    if name in original:
        s['tasks'][name] = dict(discovery=original[name]['pairs'], validation=core_validation(name),
                               target_tokens=original[name]['target_tokens'], control_tokens=original[name]['control_tokens'],
                               prompt_origin='Exact original authored PP discovery pairs, extracted without reference fields; new authored validation pairs.')
    else:
        s['tasks'][name] = dict(discovery=styled_pairs(name, discovery_requests, 'discovery'), validation=styled_pairs(name, validation_requests, 'validation'),
                               target_tokens=definitions[name][2], control_tokens=definitions[name][3],
                               prompt_origin='Behavior instructions authored from the labels in this chat; no inventory lookup.')
(P / 'protocol.json').write_text(json.dumps(s, indent=2) + '\n')
print('Prepared', len(s['tasks']), 'categories without coordinate inputs.')
