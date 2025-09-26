"""
Step 3: Analyst — Extract findings & contradictions
"""

import re

_SENT_SPLIT = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z])')
NUM_RE = re.compile(r'(\d+(?:\.\d+)?%?)')

def split_sentences(txt): return _SENT_SPLIT.split(txt) if txt else []
def extract_numbers(s): return NUM_RE.findall(s)

def analyst(results):
    findings, contradictions, nums_by_subq = [], [], {}
    for r in results:
        sid, url, text = r["subq_id"], r["url"], r["text"]
        for sent in split_sentences(text)[:5]:
            nums = extract_numbers(sent)
            if sent and len(sent) > 40:
                findings.append({"subq_id":sid,"claim":sent,"url":url,"nums":nums})
                if nums:
                    nums_by_subq.setdefault(sid, []).extend(
                        [(float(n.strip('%')), url) for n in nums if n.replace('.','',1).isdigit()]
                    )
    for sid, vals in nums_by_subq.items():
        if len(vals)>1:
            values = [v for v,u in vals]
            if max(values)-min(values) > 0.15*max(values):
                contradictions.append({"subq_id":sid,"values":vals})
    return findings, contradictions

if __name__ == "__main__":
    # Dummy test
    dummy = [{"subq_id":"Q1","url":"x","text":"H1B approval rate is 15%. Another report says 25%."}]
    f,c = analyst(dummy)
    print(f"Findings: {f}\nContradictions: {c}")
