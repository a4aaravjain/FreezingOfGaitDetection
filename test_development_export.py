import unittest,tempfile,csv
from pathlib import Path
from unittest.mock import patch
import export_development_only as module
class ExportTests(unittest.TestCase):
 def test_only_allowed_rows_are_parsed_and_exported(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d)/'source.csv';output=Path(d)/'out.csv';source.write_bytes(b'value,fog,subjectID,sessionID,taskID\n1,0,1,1,1\n"POISON UNPARSEABLE,not-a-label,19,1,1\n2,1,18,1,1\nNO_SENSOR_PARSE,NO_LABEL_PARSE,22,1,1\n')
   original=module.parse_allowed;seen=[]
   def parse(raw,columns):seen.append(raw);return original(raw,columns)
   with patch.object(module,'parse_allowed',side_effect=parse):report=module.export(source,output)
   self.assertEqual(len(seen),2);self.assertEqual(report['permitted_rows'],2)
   with output.open() as f:rows=list(csv.DictReader(f))
   self.assertEqual({int(r['subjectID']) for r in rows},{1,18});self.assertNotIn(b'POISON',output.read_bytes())
 def test_verified_real_export_has_only_allowed_participants(self):
  import json,hashlib
  p=Path(__file__).resolve().parent/'development_only_20261111_01.csv';m=json.loads(p.with_suffix('.manifest.json').read_text());self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),m['output_sha256'])
  with p.open(newline='',encoding='utf-8') as f:
   rows=csv.DictReader(f);count=0;seen=set()
   for row in rows:
    subject=int(row['subjectID']);self.assertIn(subject,module.ALLOWED);seen.add(subject);count+=1
  self.assertEqual(count,m['permitted_rows']);self.assertEqual(seen,set(module.ALLOWED))
 def test_existing_output_is_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'s').write_text('x,subjectID,sessionID,taskID\n1,1,1,1\n');(p/'o').write_text('keep')
   with self.assertRaises(FileExistsError):module.export(p/'s',p/'o')
   self.assertEqual((p/'o').read_text(),'keep')
if __name__=='__main__':unittest.main()
