"""Optional local verification; raw cache is deliberately not distributed."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-cache',type=Path,required=True);args=ap.parse_args()
    raw=args.raw_cache.read_bytes();records=[json.loads(x) for x in raw.decode().splitlines()]
    summary=json.loads((ROOT/'data/cache_summary.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==summary['cache_sha256']
    edges=pd.read_csv(ROOT/'data/cache_edges.csv');errs=[]
    for rec in records:
        outputs={(o['i'],o['j']):o for o in rec['outputs']}
        assert len(outputs)==12
        for t in edges[edges.id==rec['id']].itertuples():
            f=outputs[t.i,t.j];rev=outputs[t.j,t.i]
            errs.extend([abs(expit(f['logitA']-f['logitB'])-t.forward_p_i),
                         abs(expit(rev['logitB']-rev['logitA'])-t.reversed_mapped_p_i)])
    assert max(errs)<1e-14
    result=dict(raw_cache_sha256=summary['cache_sha256'],prompts=len(records),
        directed_outputs=sum(len(r['outputs']) for r in records),maximum_numeric_table_error=max(errs),
        raw_model_inference_repeated=False,raw_cache_distributed=False)
    (ROOT/'build').mkdir(exist_ok=True)
    (ROOT/'build/raw_cache_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
