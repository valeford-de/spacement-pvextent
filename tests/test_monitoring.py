# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Read-only live state, truthful run states and bounded in-flight log reads."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pvextent.monitoring import snapshot, _events


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def log(self, name, events):
        path = self.root / 'runs' / name / 'telemetry.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(e)+'\n' for e in events))
        return path

    def test_current_phase_counts_only_completed_actual_checks_without_loading_rasters(self):
        sample = {'full_image_inspected': True, 'proposal_overlay_inspected': True}
        self.write('artifacts/nrw-training-bulk-mask-reviews-v1/lane-a/batch-0001.review.json',
                   {'review_schema_version': '1.1', 'samples': [sample]*25})
        self.write('artifacts/nrw-training-bulk-mask-checks-v2/lane-a/batch-0001.review.json',
                   {'review_schema_version': '1.2', 'samples': [sample]*25})
        self.write('artifacts/nrw-training-bulk-mask-checks-v2/lane-b/batch-0001.review.json',
                   {'review_schema_version': '1.2', 'samples': [{**sample,'proposal_overlay_inspected':False}]*25})
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.suffix in ('.png', '.pt', '.safetensors'):
                raise AssertionError('Monitoring must not open pixels or weights')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            state = snapshot(self.root)
        self.assertEqual(state['phase'], 'mask_checks')
        self.assertEqual([state[k]['images'] for k in ('drafts','check_c','check_d')],[25,25,0])
        self.assertFalse(state['new_training_running'])
        self.assertIsNone(state['selected_run'])

    def test_live_log_is_detected_without_a_checkpoint_or_any_model_import(self):
        self.log('.new-baseline-test', [{'event':'run_started','pid':os.getpid(),'device':'cpu'},
                 {'event':'train_batch','pid':os.getpid(),'epoch':1,'global_step':20,'loss':.45}])
        state = snapshot(self.root)
        self.assertTrue(state['new_training_running'])
        self.assertEqual(state['selected_run']['loss'],[{'step':20,'value':.45}])
        self.assertEqual(state['selected_run']['device'],'cpu')

    def test_completed_history_is_never_presented_as_live_and_can_be_selected(self):
        self.write('runs/old/baseline-report.json', {'ok':True,'total_epochs':3,'global_step':123})
        path=self.root/'runs/old/history.jsonl'
        path.write_text(json.dumps({'event':'epoch','epoch':3,'global_step':123,'train_loss':.2,
                                   'validation':{'iou':.9,'dice':.95}})+'\n')
        self.assertIsNone(snapshot(self.root)['selected_run'])
        state=snapshot(self.root,'old')
        self.assertFalse(state['new_training_running'])
        self.assertEqual(state['selected_run']['state'],'completed')
        self.assertEqual(state['selected_run']['validation'][0]['iou'],.9)
        self.assertIsNone(snapshot(self.root,'../../secret')['selected_run'])

    def test_failed_run_is_not_live_even_if_its_process_still_exists(self):
        self.log('failed', [{'event':'run_started','pid':os.getpid()},
                            {'event':'run_failed','pid':os.getpid()}])
        self.assertFalse(snapshot(self.root)['new_training_running'])

    def test_partial_lines_and_large_logs_keep_actual_tail_and_initial_pid(self):
        path=self.log('tail',[{'event':'run_started','pid':os.getpid()},
                              *[{'event':'train_batch','global_step':n,'loss':.2} for n in range(100)]])
        with path.open('ab') as stream:stream.write(b'{"event":')
        with patch('pvextent.monitoring.MAX_LOG_BYTES',256):
            events=_events(path)
        self.assertEqual(events[0]['event'],'run_started')
        self.assertEqual(events[-1]['global_step'],99)
        self.assertLess(len(events),10)


if __name__ == '__main__':
    unittest.main()
