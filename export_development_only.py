"""Decode only subjectID before copying a complete allowed row. No reserved-value parsing."""
from pathlib import Path
import csv,mmap,hashlib,json,shutil,argparse
ALLOWED=tuple(range(1,19))
def parse_allowed(raw,columns):
 row=next(csv.reader([raw.decode('utf-8').rstrip('\r\n')]))
 if len(row)!=len(columns) or int(row[columns.index('subjectID')]) not in ALLOWED:raise ValueError('Invalid permitted row')
 return row

def export(source,destination,record_source_hash=False):
 source=Path(source);destination=Path(destination);count=0
 source_digest=hashlib.sha256() if record_source_hash else None
 with source.open('rb') as f:
  header=f.readline();columns=next(csv.reader([header.decode('utf-8-sig').strip()]))
  if columns[-3:]!=['subjectID','sessionID','taskID']:raise ValueError('Unsupported metadata layout; do not fall back to full-row parsing')
  with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data,destination.open('xb') as out:
   out.write(header);start=len(header)
   if source_digest is not None:source_digest.update(header)
   while start<len(data):
    newline=data.find(b'\n',start);end=len(data) if newline<0 else newline+1
    # Only delimiter positions and the subjectID substring are inspected.
    right=end
    for _ in range(2):right=data.rfind(b',',start,right)
    left=data.rfind(b',',start,right)
    if left<start or right<0:raise ValueError('Cannot locate participant metadata')
    participant=int(data[left+1:right].strip().strip(b'"'))
    # The current user explicitly requested an input hash. Hash raw bytes only;
    # no reserved row is decoded, copied into a retained row, or label-counted.
    if source_digest is not None:source_digest.update(memoryview(data)[start:end])
    if participant in ALLOWED:
     row=data[start:end];parse_allowed(row,columns);out.write(row);count+=1
    start=end
 return dict(allowed_participants=list(ALLOWED),permitted_rows=count,columns=columns,output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),reserved='Rows excluded without decoding sensor or label fields',source_content_hash=source_digest.hexdigest() if source_digest is not None else None)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--record-input-sha256',action='store_true');a=p.parse_args();report=export(a.source,a.output,a.record_input_sha256);report['command']='python export_development_only.py --source '+str(a.source)+' --output '+str(a.output)+(' --record-input-sha256' if a.record_input_sha256 else '');report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();a.output.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2));shutil.copy2(__file__,a.output.with_suffix('.export_script.py'));print(json.dumps(report,indent=2))
