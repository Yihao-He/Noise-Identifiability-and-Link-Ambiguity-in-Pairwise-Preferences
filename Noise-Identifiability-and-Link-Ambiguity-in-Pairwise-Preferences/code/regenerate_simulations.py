"""Opt-in rerun of the historical CPU experiment in a NEW output directory."""
from pathlib import Path
import argparse,shutil,subprocess,sys

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-fits',action='store_true',help='Also rerun 3,200 datasets and 6,400 model fits (CPU intensive).')
    args=p.parse_args();out=args.output.resolve()
    if out.exists():raise SystemExit('Output must be a new directory, to preserve saved results.')
    (out/'source').mkdir(parents=True);(out/'results').mkdir()
    for src in (Path(__file__).parent/'original').glob('*.py'):shutil.copy2(src,out/'source'/src.name)
    subprocess.run([sys.executable,str(out/'source/high_precision.py')],check=True)
    if args.include_fits:subprocess.run([sys.executable,str(out/'source/finite.py')],check=True)

if __name__=='__main__':main()
