"""Read-only provenance audits for the fixed published cascade run."""
from pathlib import Path
import json,unittest
from fog_pipeline import sha256

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/published_cascade_20261111_01'


class ProvenanceTests(unittest.TestCase):
    def test_fixed_partitions_exclude_each_outer_person_everywhere(self):
        p=json.loads((OUT/'preregistered_plan.json').read_text());outer_all=[]
        for f in p['folds']:
            groups=[set(f[k]) for k in ['fit','score_calibration','policy_calibration','outer']]
            for i,g in enumerate(groups):
                for h in groups[:i]:self.assertFalse(g&h)
            self.assertEqual(set.union(*groups),set(range(1,19)))
            held=f['upstream_crossfit_held'];self.assertFalse(set(held[0])&set(held[1]));self.assertEqual(set(sum(held,[])),groups[0])
            outer_all+=f['outer']
        self.assertEqual(sorted(outer_all),list(range(1,19)))

    def test_actual_upstream_artifacts_use_only_recorded_fit_people(self):
        for path in OUT.glob('fold_*/*/provenance.json'):
            p=json.loads(path.read_text());fit=set(p['fit_subjects']);excluded=set(p['held_subjects'])|set(p['outer_subjects'])|set(p['calibration_subjects'])
            self.assertFalse(fit&excluded);self.assertTrue(fit<=set(range(1,19)))
            for key in ['normalization_subjects','pretraining_subjects','hard_negative_selection_subjects']:self.assertEqual(set(p[key]),fit)
            for model in path.parent.glob('*_DONE.json'):
                m=json.loads(model.read_text());self.assertEqual(set(m['training_subjects']),fit)
            done=path.parent/'DONE.json'
            if done.exists():
                manifest=json.loads(done.read_text());self.assertTrue(all(sha256(path.parent/n)==h for n,h in manifest['artifacts'].items()))

    def test_verifier_fit_scope_and_frozen_order(self):
        for folder in OUT.glob('fold_*'):
            if not (folder/'partitions.json').exists():continue
            spec=json.loads((folder/'partitions.json').read_text());allowed=set(spec['fit'])
            for path in list(folder.glob('verifiers/*/FIT.json'))+list(folder.glob('ablations/*/FIT.json')):
                info=json.loads(path.read_text());self.assertTrue(set(info['subjects'])<=allowed)
            frozen=folder/'FROZEN_CASCADE.json';scores=folder/'outer_score_ledger.csv'
            if scores.exists():
                self.assertTrue(frozen.exists());self.assertLessEqual(frozen.stat().st_mtime,scores.stat().st_mtime)
                manifest=json.loads(frozen.read_text());self.assertTrue(all(sha256(folder/n)==h for n,h in manifest['model_hashes'].items()))


if __name__=='__main__':unittest.main()
