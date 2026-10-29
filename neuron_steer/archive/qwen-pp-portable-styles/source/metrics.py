"""Frozen deterministic content/format gates; semantic style requires sealed review."""
import re


def active(schedule, step):
    if schedule == 'continuous': return True
    if schedule == 'pulse4': return step < 4
    raise ValueError(schedule)


def measure(text, item, capped):
    if 'expected' in item:
        return {'exact_correct': text.strip() == item['expected'], 'capped': capped}
    words = re.findall(r"\b[\w]+(?:['’\-][\w]+)*\b", text)
    sentences = len(re.findall(r'[.!?]+(?=\s|$)', text.strip()))
    present = {x: bool(re.search(r'(?<!\w)' + re.escape(x) + r'(?!\w)', text, re.I)) for x in item['required_terms']}
    return {'required_terms': present, 'all_terms': all(present.values()), 'word_count': len(words), 'sentence_count': sentences,
            'format_pass': len(words) <= item['word_limit'] and sentences == item['sentence_count'], 'capped': capped}


def joint(generation, grade):
    m = generation['metrics']
    return grade['style'] == 2 and grade['quality'] == 2 and m['all_terms'] and m['format_pass'] and not m['capped']
