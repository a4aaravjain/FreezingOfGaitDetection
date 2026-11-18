"""Single approved development input; reject a changed file before parsing rows."""
from pathlib import Path
import pandas as pd
from fog_pipeline import sha256

EXPORT=Path(__file__).resolve().parent/'development_only_20261111_01.csv'
EXPECTED_SHA256='2925d016fa4fe7993b4770d9e5f7927b13b0a0f097571f14c93207078121c08c'
ALLOWED=frozenset(range(1,19))


def load_development(subjects):
    ids=set(subjects)
    if not ids or not ids.issubset(ALLOWED):
        raise ValueError('Only approved development participants1-18 may be requested')
    if sha256(EXPORT)!=EXPECTED_SHA256:
        raise ValueError('Development export checksum mismatch; no rows parsed')
    frame=pd.read_csv(EXPORT)
    if set(frame.subjectID.unique())!=ALLOWED:
        raise ValueError('Approved export participant scope mismatch')
    return frame[frame.subjectID.isin(ids)].copy()
