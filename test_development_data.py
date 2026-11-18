import unittest
from unittest.mock import patch
import development_data as module


class DevelopmentInputTests(unittest.TestCase):
    def test_outside_participant_rejected_before_file_read(self):
        with patch.object(module,'sha256') as digest,patch.object(module.pd,'read_csv') as parse:
            with self.assertRaises(ValueError):module.load_development([19])
            digest.assert_not_called();parse.assert_not_called()

    def test_changed_export_rejected_before_row_parse(self):
        with patch.object(module,'sha256',return_value='incorrect'),patch.object(module.pd,'read_csv') as parse:
            with self.assertRaises(ValueError):module.load_development([1])
            parse.assert_not_called()

    def test_ensemble_cli_routes_to_continuous_development(self):
        import ankles
        with patch('sys.argv',['ankles.py','ensemble','--output','unused_test_output']),patch('continuous_session_experiment.main') as run:
            ankles.main()
            self.assertEqual(str(run.call_args.kwargs['output']),'unused_test_output')
            self.assertEqual(run.call_args.kwargs['model_run'].name,'right_ankle_waveform_v2')


if __name__=='__main__':unittest.main()
