#!/usr/bin/env python3
"""Redaction-detector prototype: measure precision/recall pressure on the real corpus."""
import re, json, glob, os, collections, math

PATTERNS = [
 ("aws_access_key",   re.compile(r"AKIA[0-9A-Z]{16}")),
 ("github_pat",       re.compile(r"gh[posru]_[A-Za-z0-9]{30,}")),
 ("anthropic_key",    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
 ("openai_key",       re.compile(r"\bsk-[A-Za-z0-9]{32,}")),
 ("slack_token",      re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
 ("jwt",              re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.")),
 ("private_key",      re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
 ("bearer",           re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}")),
 ("pg_url_password",  re.compile(r"postgres(?:ql)?://[^:/\s]+:[^@\s]{3,}@")),
 ("assignment",       re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[=:]\s*[\"']?([^\s\"',}]{8,})")),
]
# Fields known to hold opaque-but-not-secret model data. Exercises false positives.
OPAQUE_FIELDS = ("signature", "encrypted_content")

def entropy(s):
    if not s: return 0.0
    c = collections.Counter(s)
    return -sum(n/len(s)*math.log2(n/len(s)) for n in c.values())

B64 = re.compile(r"[A-Za-z0-9+/=_\-]{40,}")

hits = collections.Counter(); lines_flagged = 0; total = 0
samples = collections.defaultdict(list)
entropy_only = 0; opaque_lines = 0

for root, pat in (("~/.claude/projects","**/*.jsonl"), ("~/.codex/sessions","**/*.jsonl")):
    os.chdir(os.path.expanduser(root))
    for f in glob.glob(pat, recursive=True):
        for line in open(f, errors="replace"):
            total += 1
            flagged = False
            for name, rx in PATTERNS:
                m = rx.search(line)
                if m:
                    hits[name] += 1; flagged = True
                    if len(samples[name]) < 3:
                        samples[name].append(m.group(0)[:70])
            if flagged: lines_flagged += 1
            else:
                # would a pure high-entropy rule add anything (and at what cost)?
                is_opaque = any(fld in line for fld in OPAQUE_FIELDS)
                for b in B64.findall(line)[:3]:
                    if entropy(b) > 4.5:
                        entropy_only += 1
                        if is_opaque: opaque_lines += 1
                        break

print(f"records scanned: {total:,}")
print(f"records flagged by named patterns: {lines_flagged:,} ({100*lines_flagged/total:.3f}%)\n")
print(f"{'pattern':20}{'hits':>8}  sample")
for name, _ in PATTERNS:
    s = samples[name][0] if samples[name] else ""
    print(f"{name:20}{hits[name]:>8}  {s}")
print(f"\nhigh-entropy-only additions: {entropy_only:,}")
print(f"  ...of which sit on records containing {OPAQUE_FIELDS}: {opaque_lines:,} "
      f"({100*opaque_lines/max(entropy_only,1):.1f}%) <- these are model signatures/ciphertext, NOT secrets")
