#!/usr/bin/env python3
"""Reconstruct continuous MLSP 0.1s segments and export 16k mono WAVs.

Uses only Python stdlib plus ffmpeg/ffprobe. Outputs a separate MLSP CSV.
No original files are modified. Each chunk inherits its source session group.
"""
from __future__ import annotations
import argparse, csv, re, subprocess, tempfile, wave
from collections import defaultdict, Counter
from pathlib import Path

PATTERN = re.compile(r'^(?P<timestamp>\d{13,14})(?P<token>MINI|PRO4|XXXX)(?P<metadata>[\d-]+)R(?P<flight>\d{6})-(?P<session>[A-Za-z]\d{3})-(?P<sequence>\d+)\.wav$', re.I)
FIELDS = ['file_path','label','source','group_id','original_path','original_sr','original_channels','duration_sec','split','notes']

def inspect(path):
    """Return (sample count, kind); preserve valid nonstandard WAV durations.

    A regular 0.1-second segment is ~9600 samples. Longer segments are
    isolated rather than joined to neighboring files (overlap unknown).
    """
    with wave.open(str(path), 'rb') as w:
        if w.getcomptype() != 'NONE':
            raise ValueError('compressed WAV')
        sr, channels = w.getframerate(), w.getnchannels()
        count, width = w.getnframes(), w.getsampwidth()
    if sr != 96000 or channels != 8 or width not in (2, 3, 4) or count == 0:
        raise ValueError(f'invalid header: sr={sr} ch={channels} width={width} frames={count}')
    if 9500 <= count <= 9700:
        return count, 'regular'
    if count >= 48000:  # at least 0.5 second usable for the existing Conformer
        return count, 'standalone'
    raise ValueError(f'nonstandard short fragment: {count} samples ({count/sr:.3f}s)')

def run_ffmpeg(files, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    # concat demuxer avoids one Python/FFmpeg process per 0.1-second input
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        listing = td / 'concat.txt'
        with listing.open('w', encoding='utf-8') as f:
            for p in files:
                absolute = p.resolve().as_posix().replace("'", "'\\''")
                f.write("file '" + absolute + "'\n")
        tmp = td / 'converted.wav'
        cmd = ['ffmpeg','-hide_banner','-nostdin','-v','error','-y',
               '-f','concat','-safe','0','-i',str(listing),
               '-map','0:a:0','-ac','1','-ar','16000','-c:a','pcm_s16le',str(tmp)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stderr[-1000:])
        with wave.open(str(tmp),'rb') as w:
            if (w.getframerate(),w.getnchannels(),w.getsampwidth()) != (16000,1,2):
                raise ValueError('incorrect converted format')
            duration=w.getnframes()/16000
        if not duration:
            raise ValueError('zero-length output')
        tmp.replace(dest)
    return duration

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('data/raw/MLSP_2022_Real_Data'))
    ap.add_argument('--output',type=Path,default=Path('data/processed/mlsp'))
    ap.add_argument('--manifest',type=Path,default=Path('data/mlsp_manifest.csv'))
    ap.add_argument('--chunk-seconds',type=float,default=30.0)
    ap.add_argument('--max-outputs',type=int,default=0,help='For smoke test: stop after N output WAVs. 0 = all')
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    per_chunk=int(round(args.chunk_seconds/0.1))
    if per_chunk < 5: ap.error('chunk-seconds must be at least 0.5')
    files=list(args.root.rglob('*.wav'))
    if not files: ap.error('No WAV files found')
    groups=defaultdict(list)
    skipped=[]
    for p in files:
        m=PATTERN.fullmatch(p.name)
        if not m:
            skipped.append((str(p),'filename not recognized')); continue
        try: n,kind=inspect(p)
        except Exception as e:
            skipped.append((str(p),str(e))); continue
        key=(m['flight'],m['session'].upper(),m['token'].upper())
        groups[key].append((int(m['sequence']),p,n,kind))

    # Build runs of *only* regular 0.1s segments. Valid longer files
    # become standalone jobs; neither side is ever joined through them.
    # This is deliberately conservative because adjacency alone does not
    # establish the temporal relationship of the abnormal-length files.
    jobs=[]
    standalone_count=0
    for (flight,session,token),segments in sorted(groups.items()):
        segments.sort(key=lambda s:s[0])
        runs=[]; run=[]; previous=None
        group_id=f'mlsp_{flight}_{session}'
        label=0 if token=='XXXX' else 1
        for item in segments:
            seq,p,n,kind=item
            if kind == 'standalone':
                if run: runs.append(run); run=[]
                filename=f'{flight}_{session}_{token}_standalone_s{seq:06d}.wav'
                dest=args.output/('drone' if label else 'background')/filename
                jobs.append(([item],dest,group_id,label,token,'standalone'))
                standalone_count += 1
                previous=seq
                continue
            if previous is not None and seq != previous+1:
                if run: runs.append(run)
                run=[]
            run.append(item)
            previous=seq
        if run: runs.append(run)
        for ri,one_run in enumerate(runs):
            for start in range(0,len(one_run),per_chunk):
                batch=one_run[start:start+per_chunk]
                if len(batch)<5:
                    skipped.extend((str(x[1]),'fragment shorter than five 0.1-second segments') for x in batch)
                    continue
                filename=f'{flight}_{session}_{token}_r{ri:02d}_s{batch[0][0]:06d}_e{batch[-1][0]:06d}.wav'
                dest=args.output/('drone' if label else 'background')/filename
                jobs.append((batch,dest,group_id,label,token,'regular'))
    jobs.sort(key=lambda job: (job[2], job[0][0][0]))
    print(f'Inputs: {len(files)} | valid groups: {len(groups)} | output chunks planned: {len(jobs)} | standalone long recordings: {standalone_count} | skipped: {len(skipped)}',flush=True)
    completed=[]
    for i,(batch,dest,group,label,token,kind) in enumerate(jobs,1):
        try:
            if args.overwrite or not dest.exists():
                duration=run_ffmpeg([x[1] for x in batch],dest)
            else:
                with wave.open(str(dest),'rb') as w: duration=w.getnframes()/w.getframerate()
            completed.append(dict(file_path=str(dest.resolve()),label=label,source='mlsp',group_id=group,
                 original_path=f'{batch[0][1]} .. {batch[-1][1]}',original_sr=96000,original_channels=8,
                 duration_sec=round(duration,5),split='',notes=(f'{token}; {len(batch)} contiguous 0.1s segments; original retained' if kind=='regular' else f'{token}; standalone {batch[0][2]/96000:.3f}s recording; continuity/overlap not verified; original retained')))
        except Exception as e:
            skipped.append((str(dest),f'conversion error: {e}'))
        if i%20==0 or i==len(jobs): print(f'Processed {i}/{len(jobs)} output chunks',flush=True)
        if args.max_outputs and i>=args.max_outputs: break
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    # Existing rows retained for resume; do not touch data/manifest.csv.
    merged={}
    if args.manifest.exists():
        with args.manifest.open(newline='',encoding='utf-8') as f:
            merged={r['file_path']:r for r in csv.DictReader(f)}
    merged.update({r['file_path']:r for r in completed})
    with args.manifest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(merged.values())
    review=args.manifest.with_name('mlsp_review.csv')
    with review.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['path','reason']);w.writerows(skipped)
    print(f'Wrote {len(completed)} output entries; manifest: {args.manifest}; review: {review}')
    print('IMPORTANT: Group split by group_id before training; never random-split reconstructed chunks.')

if __name__=='__main__': main()
