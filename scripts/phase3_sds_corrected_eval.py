"""Guarded Phase 3 SDS evaluator; never overwrites outputs or guesses ontology."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=PROJECT.parent
SDS=WORKSPACE/'strawberry-DS'/'test'
SDS_YAML=WORKSPACE/'strawberry-DS'/'data.yaml'
EXPECTED_MANIFEST='17b5a0e107d71b6dd20e6717ff5febc1c6211cdb8a0b16b25a16c3854db114a2'
MAP={0:2,1:1,2:2,3:0,4:2,5:1}
IMAGE_SUFFIXES={'.jpg','.jpeg','.png','.bmp'}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def manifest(root:Path):
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES|{'.txt'})
    h=hashlib.sha256()
    for p in files: h.update(f"{p.relative_to(root).as_posix()}\t{sha(p)}\n".encode())
    return h.hexdigest()
def mapped_class(native:int, ontology:str)->int:
    if ontology=='native_six_class':
        if native not in MAP: raise ValueError(f'illegal native SDS class {native}')
        return MAP[native]
    if ontology=='mapped_three_class':
        if native not in (0,1,2): raise ValueError(f'illegal mapped class {native}')
        return native
    raise ValueError(f'unknown ontology {ontology}')
def identity(root:Path=SDS, ontology:str='native_six_class'):
    images=[p for p in (root/'images').iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]; labels=list((root/'labels').glob('*.txt')); counts=[0,0,0]; total=0
    for p in labels:
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip(): counts[mapped_class(int(float(line.split()[0])),ontology)]+=1; total+=1
    out={'images':len(images),'labels':len(labels),'total_gt':total,'gt_fully_green_half':counts,'manifest_sha256':manifest(root)}
    expected={'images':26,'labels':26,'total_gt':104,'gt_fully_green_half':[19,78,7],'manifest_sha256':EXPECTED_MANIFEST}
    if out!=expected: raise RuntimeError(f'SDS identity mismatch: actual={out}, expected={expected}')
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--detector',type=Path,required=True);p.add_argument('--expected-checkpoint-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model-name',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--label-ontology',choices=('native_six_class','mapped_three_class'),required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if a.label_ontology!='native_six_class': raise RuntimeError('Correct frozen SDS test is native_six_class; mapped_three_class is forbidden for this entry')
    checkpoint_sha256=sha(a.detector)
    if checkpoint_sha256 != a.expected_checkpoint_sha256.lower(): raise RuntimeError(f'checkpoint SHA256 mismatch: actual={checkpoint_sha256}, expected={a.expected_checkpoint_sha256.lower()}')
    ident=identity(ontology=a.label_ontology)
    sys.path.insert(0,str(PROJECT/'scripts')); import evaluate_d3_end_to_end as ev
    records,elapsed,dropped=ev.predict_dataset('d0',a.detector,None,SDS/'images',SDS/'labels','strawberry_ds','0',0.001,0.7,640,8,64)
    metrics=ev.calculate_metrics(records,0.25)
    actual=[metrics['per_class'][n]['gt'] for n in ('fully_ripened','green','half_ripened')]
    if actual!=ident['gt_fully_green_half']: raise RuntimeError(f'evaluator GT mismatch: {actual}')
    out={'repair_status':'CORRECTED_NATIVE_SDS_TEST','model_name':a.model_name,'seed':a.seed,'detector':str(a.detector.resolve()),'checkpoint_sha256':checkpoint_sha256,'evaluated_from':{'images':str((SDS/'images').resolve()),'labels':str((SDS/'labels').resolve()),'dataset_yaml':str(SDS_YAML.resolve()),'dataset_yaml_sha256':sha(SDS_YAML)},'dataset_manifest_sha256':ident['manifest_sha256'],'dataset_identity':ident,'label_ontology':a.label_ontology,'class_order':{'0':'fully_ripened','1':'green','2':'half_ripened'},'parameters':{'system':'d0','detector_conf':0.001,'nms_iou':0.7,'imgsz':640,'batch':8,'score_threshold':0.25,'match_iou':0.5},'script_version':'phase3_sds_corrected_eval_v2','script_sha256':sha(Path(__file__)),'executed_at_utc':datetime.now(timezone.utc).isoformat(),'metrics':metrics,'elapsed_seconds':elapsed,'degenerate_predictions_dropped':dropped}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
