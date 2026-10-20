# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Synthetic cache verification; no network, imagery or model artifacts."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from pvextent.data import acquisition_cache as cache

URL = "https://www.wms.nrw.de/geobasis/wms_nw_dop?REQUEST=GetMap&BBOX=1,2,3,4"
BUILDING_URL = "https://ogc-api.nrw.de/lika/v1/collections/gebaeude_bauwerk/items?f=json&limit=500"


class AcquisitionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = self.root / "cache"
        self.fetcher = Mock(return_value=(b"synthetic-response", {"content-type": "application/octet-stream", "etag": "fixture"}))

    def tearDown(self):
        self.temporary.cleanup()

    def call(self, url=URL, maximum=1024, **kwargs):
        return cache.cached_fetch(url, maximum, cache_dir=kwargs.pop("cache_dir", self.directory),
                                  fetcher=kwargs.pop("fetcher", self.fetcher), **kwargs)

    def entry(self, url=URL):
        return self.directory / hashlib.sha256(url.encode()).hexdigest()

    def metadata(self):
        return json.loads((self.entry() / "metadata.json").read_text())

    def rewrite_metadata(self, **changes):
        value = self.metadata(); value.update(changes)
        (self.entry() / "metadata.json").write_text(json.dumps(value))

    def failures(self):
        return list((self.directory / "failures").glob("*.json"))

    def test_fetch_then_hit_preserves_original_bytes_headers_and_retrieval_time(self):
        first = self.call()
        second = self.call(fetcher=Mock(side_effect=AssertionError("No refetch")))
        self.fetcher.assert_called_once_with(URL, 1024)
        self.assertEqual(first[:2], second[:2])
        self.assertEqual(first[0], b"synthetic-response")
        self.assertFalse(first[2]["cache_hit"])
        self.assertTrue(first[2]["network_request_performed"])
        self.assertTrue(second[2]["cache_hit"])
        self.assertFalse(second[2]["network_request_performed"])
        for key in ("fetched_at_utc", "url_sha256", "response_sha256", "metadata_sha256"):
            self.assertEqual(first[2][key], second[2][key])
        self.assertEqual(self.metadata()["maximum_bytes_at_fetch"], 1024)
        self.assertFalse(self.failures())

    def test_full_query_url_keys_and_both_fixed_endpoints_are_supported(self):
        urls = (URL, URL + "&WIDTH=320", BUILDING_URL)
        for url in urls: self.call(url)
        self.assertEqual(self.fetcher.call_count, 3)
        self.assertEqual(len({self.entry(url) for url in urls}), 3)
        self.assertTrue(all(self.entry(url).is_dir() for url in urls))

    def test_stricter_current_byte_limit_rejects_cached_oversize_without_refetch(self):
        self.call()
        self.fetcher.reset_mock()
        with self.assertRaises(cache.AcquisitionCacheError): self.call(maximum=5)
        self.fetcher.assert_not_called()

    def test_nonofficial_credentials_fragments_ports_and_path_variants_are_rejected(self):
        urls = ("http://www.wms.nrw.de/geobasis/wms_nw_dop", "https://example.com/geobasis/wms_nw_dop",
                "https://www.wms.nrw.de/elsewhere", URL + "#fragment", URL + "\x00", "\x00" + URL,
                "https://user:secret@www.wms.nrw.de/geobasis/wms_nw_dop",
                "https://www.wms.nrw.de:443/geobasis/wms_nw_dop",
                "https://www.wms.nrw.de/geobasis/../geobasis/wms_nw_dop",
                "https://www.wms.nrw.de/geobasis/wms_nw_dop?x=white space")
        for url in urls:
            with self.subTest(url=url), self.assertRaises(cache.AcquisitionCacheError): self.call(url)
        self.fetcher.assert_not_called()
        self.assertFalse(self.directory.exists())

    def test_invalid_bounds_fail_before_cache_creation_or_network(self):
        for maximum in (False, 0, -1, 1.5, cache.MAX_RESPONSE_BYTES + 1):
            with self.subTest(maximum=maximum), self.assertRaises(cache.AcquisitionCacheError):
                self.call(maximum=maximum)
        self.fetcher.assert_not_called(); self.assertFalse(self.directory.exists())

    def test_corrupted_body_is_rejected_without_silent_download(self):
        self.call(); (self.entry() / "response.bin").write_bytes(b"changed-response!!")
        self.fetcher.reset_mock()
        with self.assertRaises(cache.AcquisitionCacheError): self.call()
        self.fetcher.assert_not_called()

    def test_metadata_url_binding_is_checked_not_merely_directory_name(self):
        self.call(); self.rewrite_metadata(url=URL + "&different=true")
        self.fetcher.reset_mock()
        with self.assertRaises(cache.AcquisitionCacheError): self.call()
        self.fetcher.assert_not_called()

    def test_invalid_metadata_timestamp_headers_and_extra_fields_are_rejected(self):
        self.call(); original=self.metadata()
        for changes in ({"fetched_at_utc": "2026-10-21T01:00:00"}, {"headers": {"x": 42}},
                        {"response_bytes": True}, {"extra": "unbound"}):
            with self.subTest(changes=changes):
                (self.entry() / "metadata.json").write_text(json.dumps({**original, **changes}))
                with self.assertRaises(cache.AcquisitionCacheError): self.call()
        self.assertEqual(self.fetcher.call_count, 1)

    def test_duplicate_json_keys_are_rejected(self):
        self.call(); path=self.entry() / "metadata.json"
        path.write_text(path.read_text().rstrip()[:-1] + ', "cache_schema_version":"1.0"}')
        with self.assertRaises(cache.AcquisitionCacheError): self.call()
        self.assertEqual(self.fetcher.call_count, 1)

    def test_symlinked_body_is_never_followed(self):
        self.call(); body=self.entry() / "response.bin"; body.unlink()
        outside=self.root / "outside"; outside.write_bytes(b"synthetic-response"); body.symlink_to(outside)
        with self.assertRaises(cache.AcquisitionCacheError): self.call()
        self.assertEqual(self.fetcher.call_count, 1)

    def test_hardlinked_body_is_rejected(self):
        self.call(); body=self.entry() / "response.bin"
        os.link(body, self.root / "second-link")
        with self.assertRaises(cache.AcquisitionCacheError): self.call()
        self.assertEqual(self.fetcher.call_count, 1)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO protection requires Unix")
    def test_fifo_body_is_rejected_without_blocking(self):
        self.call(); body=self.entry() / "response.bin"; body.unlink(); os.mkfifo(body)
        with self.assertRaises(cache.AcquisitionCacheError): self.call()
        self.assertEqual(self.fetcher.call_count, 1)

    def test_symlinked_entry_and_root_are_rejected_before_fetch(self):
        self.directory.mkdir(); outside=self.root / "outside"; outside.mkdir()
        self.entry().symlink_to(outside, target_is_directory=True)
        with self.assertRaises(cache.AcquisitionCacheError): self.call()
        rootlink=self.root / "root-link"; rootlink.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(cache.AcquisitionCacheError): self.call(cache_dir=rootlink)
        self.fetcher.assert_not_called(); self.assertEqual(list(outside.iterdir()), [])

    def test_fetch_failure_is_recorded_once_and_explicit_retry_can_succeed(self):
        failing=Mock(side_effect=OSError("synthetic timeout"))
        with self.assertRaises(cache.AcquisitionCacheError): self.call(fetcher=failing)
        failing.assert_called_once(); self.assertFalse(self.entry().exists())
        self.assertEqual(len(self.failures()), 1)
        self.assertEqual(json.loads(self.failures()[0].read_text())["error_type"], "OSError")
        self.call(); self.assertTrue(self.entry().is_dir())

    def test_keyboard_interrupt_does_not_poison_cache(self):
        with self.assertRaises(KeyboardInterrupt):
            self.call(fetcher=Mock(side_effect=KeyboardInterrupt()))
        self.assertFalse(self.entry().exists()); self.assertEqual(len(self.failures()), 1)
        self.call(); self.assertTrue(self.entry().is_dir())

    def test_interrupted_publication_cleans_stage_and_keeps_failure_evidence(self):
        original=cache._write_file
        def interrupted(path, payload):
            if path.name == "metadata.json": raise RuntimeError("synthetic interrupted write")
            return original(path, payload)
        with patch.object(cache, "_write_file", side_effect=interrupted), self.assertRaises(cache.AcquisitionCacheError):
            self.call()
        self.assertFalse(self.entry().exists())
        self.assertEqual(list(self.directory.glob(".*-response-*")), [])
        self.assertEqual(len(self.failures()), 1)
        self.call(); self.assertTrue(self.entry().is_dir())

    def test_interrupt_immediately_after_atomic_rename_keeps_a_verified_entry(self):
        original = cache.os.rename
        def interrupt_after_publication(source, target):
            original(source, target)
            if Path(target).resolve() == self.entry().resolve(): raise KeyboardInterrupt()
        with patch.object(cache.os, "rename", side_effect=interrupt_after_publication), self.assertRaises(KeyboardInterrupt):
            self.call()
        self.assertTrue(self.entry().is_dir())
        self.assertEqual(len(self.failures()), 1)
        result=self.call(fetcher=Mock(side_effect=AssertionError("Already published")))
        self.assertTrue(result[2]["cache_hit"])
        self.assertFalse(result[2]["network_request_performed"])

    def test_oversized_metadata_is_rejected_before_json_decoding(self):
        self.call(); (self.entry() / "metadata.json").write_bytes(b" " * (cache.MAX_METADATA_BYTES + 1))
        with patch.object(cache.json, "loads", side_effect=AssertionError("Do not decode oversized metadata")):
            with self.assertRaises(cache.AcquisitionCacheError): self.call()
        self.assertEqual(self.fetcher.call_count, 1)

    def test_oversized_invalid_body_and_headers_never_publish(self):
        responses = ((b"x" * 1025, {}), ("not bytes", {}), (b"ok", {"bad": []}),
                     (b"ok", {"long": "x" * 8193}))
        for response in responses:
            with self.subTest(response_type=type(response[0]).__name__), self.assertRaises(cache.AcquisitionCacheError):
                self.call(fetcher=Mock(return_value=response))
            self.assertFalse(self.entry().exists())
        self.assertEqual(len(self.failures()), len(responses))

    def test_concurrent_same_url_fetches_once(self):
        count=0; guard=threading.Lock()
        def fetch(url, maximum):
            nonlocal count
            with guard: count += 1
            time.sleep(.03)
            return b"concurrent fixture", {"content-type": "text/plain"}
        with ThreadPoolExecutor(max_workers=4) as workers:
            results=list(workers.map(lambda _: self.call(fetcher=fetch), range(12)))
        self.assertEqual(count, 1)
        self.assertEqual(sum(r[2]["network_request_performed"] for r in results), 1)
        self.assertEqual(sum(not r[2]["cache_hit"] for r in results), 1)
        self.assertTrue(all(r[:2] == results[0][:2] for r in results))

    def test_concurrent_process_winner_is_not_overwritten_and_network_flag_is_honest(self):
        def publish_winner_then_return_loser(url, maximum):
            winner_directory=self.root / "other-process-fixture"
            cache.cached_fetch(url, maximum, cache_dir=winner_directory,
                               fetcher=lambda *_: (b"winner", {"etag": "winner"}))
            shutil.copytree(winner_directory / self.entry().name, self.entry())
            return b"loser", {"etag": "loser"}
        payload, headers, evidence=self.call(fetcher=publish_winner_then_return_loser)
        self.assertEqual(payload, b"winner")
        self.assertEqual(headers, {"etag": "winner"})
        self.assertTrue(evidence["cache_hit"])
        self.assertTrue(evidence["network_request_performed"])
        self.assertTrue(evidence["publication_race"])
        self.assertEqual(evidence["discarded_response_sha256"], hashlib.sha256(b"loser").hexdigest())
        self.assertEqual((self.entry() / "response.bin").read_bytes(), b"winner")
        self.assertFalse(self.failures())
        self.assertEqual(list(self.directory.glob(".*-response-*")), [])

    def test_metadata_and_body_are_atomically_published_together(self):
        original=cache._write_file; observed=[]
        def inspect_stage(path, payload):
            observed.append(self.entry().exists())
            return original(path, payload)
        with patch.object(cache, "_write_file", side_effect=inspect_stage): self.call()
        self.assertEqual(observed, [False, False])
        self.assertEqual({p.name for p in self.entry().iterdir()}, {"response.bin", "metadata.json"})


if __name__ == "__main__":
    unittest.main()
