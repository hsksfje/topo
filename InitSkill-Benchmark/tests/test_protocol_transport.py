"""A returned Windows protocol must retain the SHA used by every trajectory."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from initskill.methods.benchmark_init_v1.common import protocol


class ProtocolTransportTest(unittest.TestCase):
    def test_windows_newline_identity_and_drift_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);value=protocol(folder);path=folder/'protocol.json'
            path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(protocol(folder),value)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
            changed=json.loads(path.read_text());changed['candidates_per_task']=2
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Frozen protocol changed'):protocol(folder)


if __name__=='__main__':unittest.main()
