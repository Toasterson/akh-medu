"""Contract tests for profile validation and complete manifest requirements."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('images', Path(__file__).with_name('build-platform-images.py'))
images = importlib.util.module_from_spec(spec)
spec.loader.exec_module(images)


class ProfileTests(unittest.TestCase):
    def test_counts_samples_without_counting_string_table(self):
        self.assertEqual(images.sample_count(b'\x12\x02\x08\x01\x32\x02ok\x12\x02\x08\x02'), 2)

    def test_empty_profile_has_no_samples(self):
        self.assertEqual(images.sample_count(b'\x32\x02ok'), 0)

    def test_truncated_profile_is_rejected(self):
        with self.assertRaises((ValueError, IndexError)):
            images.sample_count(b'\x12\x06short')


class ManifestTests(unittest.TestCase):
    def manifest(self, arches):
        return json.dumps({'manifests': [{'digest': 'sha256:' + arch,
                                         'platform': {'os': 'linux', 'architecture': arch}}
                                        for arch in arches]})

    def test_missing_arm_is_not_a_complete_set(self):
        with patch.object(images, 'run', return_value=self.manifest(['amd64'])):
            with self.assertRaisesRegex(RuntimeError, 'missing requested platform'):
                images.image_manifest('registry/image:main-abc', images.ARCHES)

    def test_records_exact_index_and_platform_digests(self):
        with patch.object(images, 'run', side_effect=[self.manifest(images.ARCHES), 'Name: registry/image\nDigest: sha256:index']):
            record = images.image_manifest('registry/image:main-abc', images.ARCHES)
        self.assertEqual(record['image'], 'registry/image@sha256:index')
        self.assertEqual(record['platform_digests'], {'amd64': 'sha256:amd64', 'arm64': 'sha256:arm64'})


class CollectionTests(unittest.TestCase):
    def partials(self):
        sha = 'a' * 40
        hashes = {path: 'b' * 64 for path in ['/usr/local/bin/akhrtd', '/usr/local/bin/akhexecd', '/usr/local/bin/examples/native-provider-probe']}
        receipt = {'wire': {'result': 'passed'}, 'cancel': {'result': 'passed'}, 'runtime_artifact_sha256': hashes}
        partials = {}
        for group, names in images.GROUPS.items():
            partial = {'group': group, 'complete': False, 'source_commit': sha,
                       'main_ancestry_verified': True, 'homelab_source_commit': images.HOMELAB_SHA,
                       'native_conformance': {arch: {'codex': receipt, 'claude': receipt} for arch in images.ARCHES}, 'images': {}}
            for name in names:
                arches = ['amd64'] if name in images.AMD64_ONLY else images.ARCHES
                smoke = {}
                for arch in arches:
                    proof = {'python': True, 'ssh': True} if name == 'relay' else {'binaries': sorted(images.REQUIRED_BINARIES[name])}
                    if name in images.HEAP_BINARIES:
                        proof['heap'] = [{'binary': images.HEAP_BINARIES[name], 'profile_samples': 1, 'application_symbols': True}]
                    smoke[arch] = proof
                partial['images'][name] = {'source_commit': sha, 'tag': 'registry/' + name + ':main-' + sha,
                    'index_digest': 'sha256:' + 'c' * 64, 'platform_digests': {arch: 'sha256:' + arch for arch in arches}, 'smoke': smoke}
            partials[group] = partial
        return partials, sha

    def test_partial_set_cannot_be_promoted(self):
        partials, sha = self.partials()
        del partials['core']
        with self.assertRaisesRegex(RuntimeError, 'missing image groups'):
            images.assemble_partials(partials, sha)

    def test_source_mismatch_cannot_be_promoted(self):
        partials, sha = self.partials()
        partials['daemons']['source_commit'] = 'd' * 40
        with self.assertRaisesRegex(RuntimeError, 'source or ancestry mismatch'):
            images.assemble_partials(partials, sha)

    def test_missing_heap_proof_cannot_be_promoted(self):
        partials, sha = self.partials()
        partials['daemons']['images']['sessions']['smoke']['arm64']['heap'] = []
        with self.assertRaisesRegex(RuntimeError, 'heap smoke proof missing'):
            images.assemble_partials(partials, sha)

    def test_complete_commit_matched_set_is_accepted(self):
        partials, sha = self.partials()
        result = images.assemble_partials(partials, sha)
        self.assertTrue(result['complete'])
        self.assertEqual(set(result['images']), images.EXPECTED)
        self.assertEqual(len(result['images']), 19)


class PromotionTests(unittest.TestCase):
    def evidence(self):
        return {'complete': True, 'source_commit': 'a' * 40, 'images': {
            'home-runtime': {'image': 'registry/runtime@sha256:verified',
                             'platform_digests': {'arm64': 'sha256:arm'},
                             'native_receipts_embedded': True}}}

    def test_incomplete_set_never_changes_live_channel(self):
        evidence = self.evidence()
        evidence['complete'] = False
        with patch.object(images, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'complete commit-matched'):
                images.promote_verified_set(evidence, 'a' * 40)
            run.assert_not_called()

    def test_native_receipts_required_before_promotion(self):
        evidence = self.evidence()
        evidence['images']['home-runtime'].pop('native_receipts_embedded')
        with patch.object(images, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'embedded CI receipts'):
                images.promote_verified_set(evidence, 'a' * 40)
            run.assert_not_called()

    def test_outdated_pipeline_cannot_roll_channel_back(self):
        with patch.object(images, 'released_set', return_value={'source_commit': 'b' * 40}), \
             patch.object(images, 'is_ancestor', return_value=False), \
             patch.object(images, 'run') as run:
            self.assertFalse(images.promote_verified_set(self.evidence(), 'a' * 40))
            run.assert_not_called()

    def test_verified_successor_can_promote_even_if_source_main_has_advanced(self):
        evidence = self.evidence()
        with patch.object(images, 'released_set', side_effect=[{'source_commit': 'b' * 40}, evidence]), \
             patch.object(images, 'is_ancestor', return_value=True) as ancestry, \
             patch.object(images, 'run') as run, \
             patch.object(images, 'image_manifest', return_value={'platform_digests': {'arm64': 'sha256:arm'}}):
            self.assertTrue(images.promote_verified_set(evidence, 'a' * 40))
            ancestry.assert_called_once_with('b' * 40, 'a' * 40)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args[-2:], (images.REGISTRY + '/akh-image-set:main-ready',
                images.REGISTRY + '/akh-image-set:main-' + 'a' * 40))

    def test_first_release_promotes_verified_digest_before_release_marker(self):
        evidence = self.evidence()
        with patch.object(images, 'released_set', side_effect=[None, evidence]), \
             patch.object(images, 'run') as run, \
             patch.object(images, 'image_manifest', return_value={'platform_digests': {'arm64': 'sha256:arm'}}):
            self.assertTrue(images.promote_verified_set(evidence, 'a' * 40))
            self.assertEqual(run.call_args_list[0].args, ('docker', 'buildx', 'imagetools', 'create',
                '--tag', 'registry/runtime:main-ready', 'registry/runtime@sha256:verified'))

    def test_digest_mismatch_does_not_publish_release_marker(self):
        with patch.object(images, 'released_set', return_value=None), \
             patch.object(images, 'run') as run, \
             patch.object(images, 'image_manifest', return_value={'platform_digests': {}}):
            with self.assertRaisesRegex(RuntimeError, 'channel differs'):
                images.promote_verified_set(self.evidence(), 'a' * 40)
            self.assertEqual(run.call_count, 1)

    def test_registry_error_is_not_treated_as_first_release(self):
        result = images.subprocess.CompletedProcess([], 1, '', 'unauthorized: authentication required')
        with patch.object(images.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'cannot inspect'):
                images.released_set()

    def test_missing_marker_allows_first_release(self):
        result = images.subprocess.CompletedProcess([], 1, '', 'manifest unknown')
        with patch.object(images.subprocess, 'run', return_value=result):
            self.assertIsNone(images.released_set())

    def test_malformed_marker_cannot_be_promoted_over(self):
        result = images.subprocess.CompletedProcess([], 0, '{}', '')
        with patch.object(images.subprocess, 'run', return_value=result), \
             patch.object(images, 'read_manifest', return_value={'complete': False}):
            with self.assertRaisesRegex(RuntimeError, 'invalid verified release marker'):
                images.released_set()


if __name__ == '__main__':
    unittest.main()
