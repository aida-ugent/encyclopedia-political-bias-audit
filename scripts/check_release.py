"""Check the exact Git index for private data and accidental notebook output.

Run after staging and before committing. Only explicitly reviewed source paths
are allowed. This is a guardrail, not a replacement for manual privacy review.
"""
import ast
import json
from pathlib import PurePosixPath
import re
import subprocess


def check_index():
    names = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
    problems = []
    top_level = {'.gitignore', '.env.example', 'README.md', 'LICENSE', 'requirements.txt'}
    assets = {'assets/methodology.svg', 'assets/political-bias-audit.jpeg'}
    for name in filter(None, names):
        path = PurePosixPath(name)
        allowed = (name in top_level or name in assets
                   or (path.parts[0] in {'src', 'human_study', 'scripts', 'tests'} and path.suffix == '.py')
                   or (path.parts[0] == 'notebooks' and path.suffix == '.ipynb'))
        if not allowed:
            problems.append(f'{name}: unapproved release path')
        data = subprocess.check_output(['git', 'show', ':' + name])
        if path.suffix == '.py':
            ast.parse(data, filename=name)
        if path.suffix == '.ipynb':
            notebook = json.loads(data)
            for cell in notebook['cells']:
                if cell.get('outputs') or cell.get('execution_count') is not None or cell.get('attachments'):
                    problems.append(f'{name}: notebook contains output, execution state, or attachments')
        if name in assets:
            continue  # Binary/embedded assets require manual inspection.
        text = data.decode('utf-8')
        patterns = {
            'participant-like identifier': r'\b[a-f0-9]{24}\b',
            'credential': r'(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',
            'local home path': r'/(?:home|Users)/[A-Za-z0-9_.-]+/',
        }
        for label, pattern in patterns.items():
            if re.search(pattern, text):
                problems.append(f'{name}: possible {label}')
    return problems


if __name__ == '__main__':
    findings = check_index()
    if findings:
        raise SystemExit('\n'.join(findings))
    print('Release index checks passed.')
