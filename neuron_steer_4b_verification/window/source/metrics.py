"""Frozen text gates and timing definitions; no model dependencies."""
import re

def active(schedule,step):
    return schedule=='continuous' or schedule=='pulse4' and step<4 or schedule=='prefill_only' and step==0

def exact_integer(text):
    match=re.fullmatch(r'\s*(-?\d+)\s*[.!]?\s*',text)
    return int(match.group(1)) if match else None

def measure(item,text,capped=False):
    task=item.get('task','protection')
    if task=='egg':
        words=re.findall(r"[A-Za-z]+",text.lower());pairs=list(zip(words,words[1:]))
        coherent=len(words)>=6 and len(set(pairs))/max(1,len(pairs))>=.6
        correct=bool(re.search(r'\b'+re.escape(item['required'])+r'\b',text,re.I))
        trait=bool(re.search(r'\beggs?\b',text,re.I))
        return {'correct':correct,'trait':trait,'coherent':coherent,'joint':correct and trait and coherent and not capped}
    if task=='verification':
        vals=re.findall(r'ANSWER\s*=\s*(-?\d+)\b',text,re.I)
        correct=bool(vals) and int(vals[-1])==item['expected']
        trait=bool(re.search(r'\b(check\w*|verif\w*|confirm\w*|recalcul\w*)\b',text,re.I))
        return {'correct':correct,'trait':trait,'joint':correct and trait and not capped}
    if item.get('kind','integer')=='integer':correct=exact_integer(text)==item['expected']
    else:correct=text.strip()==str(item['expected'])
    trait=task=='zero' and item['expected']==0 and correct
    return {'correct':correct,'trait':trait,'joint':trait and not capped}
