#!/usr/bin/env python3
"""Aggregate C-UAS datasets into 16 kHz mono PCM WAVs using FFmpeg.

Local imports require Python standard library + ffmpeg/ffprobe on PATH.
DADS Hugging Face import additionally requires `datasets`, `torchcodec` and `torch`.
Review label/group assumptions before training. Originals are not modified.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aac', '.aiff', '.aif'}
FIELDS = ['file_path','label','source','group_id','original_path',
          'original_sr','original_channels','duration_sec','split','notes']

def stable_id(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:14]

def command(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout).strip()[-1200:])
    return r

def probe_audio(path):
    result = command(['ffprobe','-v','error','-select_streams','a:0',
        '-show_entries','stream=sample_rate,channels:format=duration',
        '-of','json',str(path)])
    info = json.loads(result.stdout)
    streams = info.get('streams',[])
    if not streams:
        raise ValueError('No audio stream')
    stream = streams[0]
    return int(stream['sample_rate']), int(stream['channels'])

def convert_file(src, dst, overwrite=False):
    src, dst = Path(src), Path(dst)
    sr, channels = probe_audio(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if overwrite or not dst.exists():
        # Write to temporary file first so an interrupted conversion never leaves a partial target.
        with tempfile.NamedTemporaryFile(dir=dst.parent, suffix='.wav', delete=False) as f:
            tmp = Path(f.name)
        try:
            command(['ffmpeg','-hide_banner','-nostdin','-loglevel','error',
                     '-y','-i',str(src),'-map','0:a:0','-vn','-ac','1',
                     '-ar','16000','-c:a','pcm_s16le',str(tmp)])
            with wave.open(str(tmp),'rb') as wav:
                frames = wav.getnframes()
                if wav.getframerate()!=16000 or wav.getnchannels()!=1 or wav.getsampwidth()!=2 or frames<100:
                    raise ValueError('Converted file format invalid or too short')
            tmp.replace(dst)
        finally:
            tmp.unlink(missing_ok=True)
    with wave.open(str(dst),'rb') as wav:
        duration = wav.getnframes()/wav.getframerate()
    return sr, channels, duration

def export_one(src, original_id, source, label, group_id, out_root,
               original_path='', split='', notes='', overwrite=False):
    if label not in (0,1):
        raise ValueError('Label not verified')
    folder = out_root/'processed'/source/('drone' if label else 'background')
    target = folder/(stable_id(source+'|'+original_id)+'.wav')
    sr, channels, duration = convert_file(src,target,overwrite=overwrite)
    return dict(file_path=str(target.resolve()),label=label,source=source,
        group_id=group_id,original_path=original_path or str(src.resolve()),
        original_sr=sr,original_channels=channels,duration_sec=round(duration,5),
        split=split,notes=notes)

def label_from_local(path,root,rules):
    for suffix in ('_label.txt','.label.txt'):
        sidecar=path.with_name(path.stem+suffix)
        if sidecar.exists():
            value=sidecar.read_text(encoding='utf-8').strip().lower()
            if value in ('1','drone','positive','present'): return 1,'sidecar'
            if value in ('0','background','negative','absent','no_drone'): return 0,'sidecar'
            return None,f'unrecognized sidecar: {value}'
    rel=path.relative_to(root).as_posix().lower()
    matched={int(label) for label,patterns in rules.items()
             if any(re.search(pattern,rel) for pattern in patterns)}
    if len(matched)==1: return matched.pop(),'pattern rule'
    return None, 'conflicting rules' if matched else 'no explicit label rule'

def mlsp_metadata(filename):
    stem = Path(filename).stem

    pattern = (
        r"^(?P<timestamp>\d{13,14})"
        r"(?P<token>MINI|PRO4|XXXX)"
        r"(?P<metadata>[\d-]+)"
        r"R(?P<flight>\d{6})-"
        r"(?P<session>[A-Za-z]\d{3})-"
        r"(?P<sequence>\d+)$"
    )

    match = re.fullmatch(pattern, stem, re.IGNORECASE)

    if match is None:
        return None

    token = match.group("token").upper()
    flight = match.group("flight")
    session = match.group("session")

    label = 0 if token == "XXXX" else 1
    group_id = f"mlsp_{flight}_{session}"

    return label, group_id, token

def scan_directory(root,source,rules,out_root,pending,records,overwrite):
    if not root.is_dir(): raise FileNotFoundError(root)
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    print(f'[{source}] Found {len(files)} candidate audio files',flush=True)
    for idx,path in enumerate(files,1):
        rel=path.relative_to(root).as_posix()
        split=''
        if source=='mlsp':
            meta=mlsp_metadata(path.name)
            if meta is None:
                pending.append((source,str(path),'MLSP filename not recognized: manual review needed'))
                continue
            label,group,token=meta
            note=f'MLSP {token}; verify recording group'
        elif source=='x4':
            label,group=1,'x4_unverified_session'
            note='Drone-present collection; verify which windows have audible drones'
        else:
            label,reason=label_from_local(path,root,rules)
            if label is None:
                pending.append((source,str(path),reason)); continue
            group=f'{source}_{path.parent.relative_to(root).as_posix()}'
            note=f'{reason}; verify session grouping'
            low=rel.lower()
            if source=='lab':
                if any(x in low for x in ('large_holystone','holystone_large','hs720e')): split='test'
                elif any(x in low for x in ('small_holystone','holystone_small')): split='challenge'
        try:
            records.append(export_one(path,rel,source,label,group,out_root,
                original_path=str(path.resolve()),split=split,notes=note,overwrite=overwrite))
        except Exception as exc:
            pending.append((source,str(path),f'conversion error: {exc}'))
        if idx%100==0: print(f'[{source}] {idx}/{len(files)} scanned',flush=True)

def export_dads(out_root,pending,records,limit,overwrite):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('DADS requires datasets and torchcodec; install separately') from exc
    ds=load_dataset('geronimobasso/drone-audio-detection-samples',split='train')
    n=len(ds) if limit is None else min(len(ds),limit)
    print(f'[dads] Exporting {n} examples')
    for i in range(n):
        try:
            ex=ds[i]
            label=int(ex['label'])
            if label not in (0,1): raise ValueError('Unexpected label')
            audio=ex['audio']
            if not hasattr(audio,'get_all_samples'):
                raise ValueError('DADS decoder API differs; review before exporting')
            sample=audio.get_all_samples()
            signal=sample.data.detach().cpu().to(dtype=__import__('torch').float32)
            if signal.ndim!=2 or signal.shape[0]!=1:
                raise ValueError('Expected single-channel DADS signal')
            # Raw float32 PCM via built-in array, avoiding NumPy and SoundFile.
            from array import array
            values=array('f',signal[0].tolist())
            with tempfile.TemporaryDirectory() as td:
                raw=Path(td)/'audio.f32'
                raw.write_bytes(values.tobytes())
                wav=Path(td)/'audio.wav'
                command(['ffmpeg','-hide_banner','-nostdin','-loglevel','error','-y',
                         '-f','f32le','-ar',str(sample.sample_rate),'-ac','1',
                         '-i',str(raw),'-c:a','pcm_s16le',str(wav)])
                group=ex.get('recording_id') or ex.get('source_id') or ex.get('session_id') or 'dads_group_unverified'
                records.append(export_one(wav,f'dads_{i}','dads',label,str(group),out_root,
                    original_path=f'hf://dads/{i}',notes='Verify DADS labels and original recording grouping',overwrite=overwrite))
        except Exception as exc:
            pending.append(('dads',f'hf://dads/{i}',str(exc)))
        if (i+1)%100==0: print(f'[dads] {i+1}/{n}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data')
    p.add_argument('--x4')
    p.add_argument('--mlsp')
    p.add_argument('--lab')
    p.add_argument('--lab-rules')
    p.add_argument('--dads',action='store_true')
    p.add_argument('--dads-limit',type=int)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        p.error('FFmpeg and FFprobe must be on PATH')
    if not any((args.x4,args.mlsp,args.lab,args.dads)):
        p.error('Provide one or more dataset flags')
    out=Path(args.output).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    rules={}
    if args.lab_rules:
        rules=json.loads(Path(args.lab_rules).read_text(encoding='utf-8'))
        if not set(rules).issubset({'0','1'}): p.error('Lab rule keys must be "0"/"1"')
    records,pending=[],[]
    for source in ('x4','mlsp','lab'):
        path=getattr(args,source)
        if path: scan_directory(Path(path).resolve(),source,rules,out,pending,records,args.overwrite)
    if args.dads: export_dads(out,pending,records,args.dads_limit,args.overwrite)
    manifest=out/'manifest.csv'
    merged={}
    if manifest.exists():
        with manifest.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f): merged[row['file_path']]=row
    for row in records: merged[row['file_path']]=row
    with manifest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(sorted(merged.values(),key=lambda x:x['file_path']))
    review=out/'review_required.csv'
    with review.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['source','original_path','reason']);writer.writerows(pending)
    print(f'Manifest: {manifest} ({len(merged)} total entries)')
    print(f'Review: {review} ({len(pending)} items)')
    print('IMPORTANT: Verify labels, groups, and train/val/test splits before training.')

if __name__=='__main__': main()
