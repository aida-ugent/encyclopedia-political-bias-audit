import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'llm_experiments'))


class RuntimeRegressionTests(unittest.TestCase):
    def test_deepseek_accepts_messages_and_forwards_token_limit(self):
        from model_client.api_deepseek_client import ApiDeepSeekModelClient
        client = ApiDeepSeekModelClient.__new__(ApiDeepSeekModelClient)
        client.model_name = 'synthetic-model'
        client.client = MagicMock()
        client.throttle_manager = MagicMock()
        client.client.chat.completions.create.return_value.choices[0].message.content = 'ok'
        messages = [{'role': 'user', 'content': 'synthetic prompt'}]
        self.assertEqual(client.ask(message=messages, max_new_tokens=17), 'ok')
        client.client.chat.completions.create.assert_called_once_with(
            model='synthetic-model', messages=messages, max_tokens=17)
        client.throttle_manager.throttle.assert_called_once()

    def test_empty_difference_in_differences(self):
        import pandas as pd
        from src.analysis.statistical_analysis.visualisation.descriptive_stats import _compute_did_results
        result = _compute_did_results(pd.DataFrame(), [])
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ['bool_column', 'neutrality_diff'])

    def test_package_exports_exist(self):
        import src.analysis.statistical_analysis as analysis
        for name in analysis.__all__:
            self.assertTrue(hasattr(analysis, name), name)


class ReleaseGuardTests(unittest.TestCase):
    def test_staged_private_files_and_outputs_are_rejected(self):
        spec = importlib.util.spec_from_file_location('check_release', ROOT / 'scripts/check_release.py')
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            subprocess.run(['git', 'init', '-q', tmp], check=True)
            (cwd / 'README.md').write_text('Synthetic release fixture\n')
            subprocess.run(['git', '-C', tmp, 'add', '.'], check=True)
            real_check_output = subprocess.check_output
            with patch.object(guard.subprocess, 'check_output', side_effect=lambda args: real_check_output(args, cwd=tmp)):
                self.assertEqual(guard.check_index(), [])
                (cwd / 'human_study').mkdir()
                (cwd / 'human_study/excluded_participants.json').write_text('[]')
                (cwd / 'human_study/example.py').write_text('ID = "' + 'a' * 24 + '"\n')
                (cwd / 'notebooks').mkdir()
                (cwd / 'notebooks/example.ipynb').write_text('{"cells":[{"outputs":[{"text":"private"}],"execution_count":1}]}')
                subprocess.run(['git', '-C', tmp, 'add', '.'], check=True)
                problems = guard.check_index()
                self.assertTrue(any('unapproved release path' in p for p in problems))
                self.assertTrue(any('participant-like identifier' in p for p in problems))
                self.assertTrue(any('notebook contains output' in p for p in problems))


if __name__ == '__main__':
    unittest.main()
