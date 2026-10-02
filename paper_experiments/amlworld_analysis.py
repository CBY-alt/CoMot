#!/usr/bin/env python3
"""Recompute AMLWorld Fig. 6/7/8 data with the configured score and Top-1000."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.methods.comot_pipeline.candidate_reranker import candidate_type_bonus,compactness_bonus,size_penalty
from src.experiments import prioritization as rp
from src.experiments import ablation as ab
from src.evaluation.metrics import compute_recovery_metrics

OUT=ROOT/'outputs'/'paper'/'amlworld_analyses'
OUT.mkdir(parents=True,exist_ok=True)
TOPK=1000
HYBRID_PREFIX=50

def parse(v):
 x=json.loads(v) if isinstance(v,str) else v
 return x if isinstance(x,list) else []
def strip(v):return str(v).split('::',1)[-1]
def tuned_score(row):
 typ=str(row['candidate_type']).upper();nn=int(row['num_nodes']);ne=int(row['num_edges'])
 sm=float(row['score_mean']);sn=float(row['score_min']);stability=-abs(sm-sn)
 rich=0.
 if typ=='CHAIN':rich=.03*min(max(ne-2,0),4)
 elif typ=='CYCLE':rich=.03*min(max(nn-2,0),3)
 elif typ in ('FAN_IN','FAN_OUT'):rich=.01*min(max(ne-2,0),2)
 return .60*sm+.40*sn-5.00*stability+candidate_type_bonus(typ)+.10*compactness_bonus(typ,nn,ne)+rich-size_penalty(typ,nn,ne)
def load_df(path):
 fs=[path/n for n in ['chain_candidates.csv','fan_candidates.csv','cycle_candidates.csv']]
 d=pd.concat([pd.read_csv(f) for f in fs if f.exists()],ignore_index=True)
 d['variant_score']=d.apply(tuned_score,axis=1)
 return d.sort_values(['variant_score','score_mean','score_min','num_edges'],ascending=False).reset_index(drop=True)
def pred_from_df(d):
 out=[]
 for _,r in d.head(TOPK).iterrows():
  out.append({'prediction_id':str(r.candidate_id),'query_id':str(r.candidate_type),
   'predicted_nodes':[strip(x) for x in parse(r.nodes_json)],
   'predicted_edges':[str(x) for x in parse(r.tx_ids_json)],'score':float(r.variant_score),'runtime':0.})
 return out
def metric_row(m):
 p=m['overlap']['candidate_purity'];r=m['motif_level']['motif_recall'];f=2*p*r/(p+r) if p+r else 0
 return {'motif_f1':f,'edge_f1':m['edge_level']['f1'],'node_f1':m['node_level']['f1'],
         'motif_jaccard':m['motif_level']['jaccard_similarity'],'purity':p,'motif_recall':r}
def official_records(records,ground_truth):
 preds=[]
 for record in records[:TOPK]:
  preds.append({'prediction_id':record['candidate_id'],'query_id':record['query_id'],
   'predicted_nodes':list(record['nodes']),'predicted_edges':list(record['edges']),
   'score':record['score'],'runtime':0.0})
 metrics,_=compute_recovery_metrics(preds,ground_truth)
 return metric_row(metrics)
def summarize(d,groups,metrics):
 rows=[]
 for keys,z in d.groupby(groups,sort=False):
  if not isinstance(keys,tuple):keys=(keys,)
  row=dict(zip(groups,keys));row['num_seeds']=z.seed.nunique()
  for c in metrics:
   row[c+'_mean']=z[c].mean();row[c+'_std']=z[c].std(ddof=1)
  rows.append(row)
 return pd.DataFrame(rows)

# Figure 6: same pool, new Base; Diverse and Hybrid derived from that Base.
f6=[]
for seed in (0,1,2):
 run=ROOT/f'outputs/amlworld/comot/seed_{seed}'
 d,src=rp.load_aml_candidates(run)
 d['base_score']=d.apply(tuned_score,axis=1)
 order=d.sort_values(['base_score','score_mean','score_min','num_edges'],ascending=False).candidate_id.astype(str).tolist()
 rank={cid:i for i,cid in enumerate(order)};d['base_rank']=d.candidate_id.astype(str).map(rank)
 gt,edge_to,motif_to=rp.load_ground_truth('amlworld',run)
 d=rp.attach_hits(d,edge_to);types=set(d.candidate_type.astype(str))
 for strategy in rp.STRATEGIES:
  ordered,_=rp.strategy_order(strategy,d,HYBRID_PREFIX)
  m=rp.evaluate_order(ordered,gt,motif_to,types)
  f6.append({'seed':seed,'strategy':strategy,**m})
f6=pd.DataFrame(f6);f6.to_csv(OUT/'figure6_prioritization_by_seed.csv',index=False)
f6s=summarize(f6,['strategy'],rp.METRIC_COLUMNS)
f6s.to_csv(OUT/'figure6_prioritization_summary.csv',index=False)

# Figure 7: all AML ablations under Top-1000 and the shared repository evaluator.
f7=[];gt_json=json.load(open(ROOT/'data/processed/amlworld/ground_truth.json'))
for seed in (0,1,2):
 run=ROOT/f'outputs/amlworld/comot/seed_{seed}'
 raw,_=ab.load_aml_raw_pool(run)
 df=load_df(run/f'semotif_candidates_{ab.AML_TAG}')
 full=[]
 for i,r in df.iterrows():
  full.append(ab.candidate_record(r.candidate_id,r.candidate_type,r.candidate_type,parse(r.nodes_json),parse(r.tx_ids_json),r.variant_score,order=i))
 for variant in ab.VARIANTS:
  if variant=='Full CoMot': pool=full
  else: pool,_,_,_=ab.build_variant_pool('amlworld',run,variant,raw,full)
  m=official_records(pool,gt_json)
  f7.append({'seed':seed,'variant':variant,'pool_size':len(pool),'n_evaluated':min(TOPK,len(pool)),**m})
f7=pd.DataFrame(f7);f7.to_csv(OUT/'figure7_ablation_by_seed.csv',index=False)
f7s=summarize(f7,['variant'],['motif_f1','edge_f1','node_f1','motif_jaccard','purity','motif_recall','pool_size','n_evaluated'])
f7s.to_csv(OUT/'figure7_ablation_summary.csv',index=False)

# Figure 8: clean plus nine saved perturbed assembled pools, all with same score and Top-1000.
settings=['clean','missing10','missing30','missing50','spurious10','spurious30','spurious50','endpoint10','endpoint30','endpoint50']
f8=[]
for setting in settings:
 for seed in (0,1,2):
  if setting=='clean': p=ROOT/f'outputs/amlworld/comot/seed_{seed}/semotif_candidates_five_party'
  else:p=ROOT/f'outputs/boundary_robustness/amlworld/{setting}/seed_{seed}/candidates'
  d=load_df(p);pred=pred_from_df(d);m,_=compute_recovery_metrics(pred,gt_json)
  f8.append({'setting':setting,'seed':seed,'pool_size':len(d),'n_evaluated':len(pred),**metric_row(m)})
f8=pd.DataFrame(f8);f8.to_csv(OUT/'figure8_robustness_by_seed.csv',index=False)
f8s=summarize(f8,['setting'],['motif_f1','edge_f1','node_f1','motif_jaccard','purity','motif_recall','pool_size','n_evaluated'])
f8s.to_csv(OUT/'figure8_robustness_summary.csv',index=False)

print('\nFIGURE 6 AMLWORLD')
print(f6s.to_string(index=False))
print('\nFIGURE 7 AMLWORLD')
print(f7s.to_string(index=False))
print('\nFIGURE 8 AMLWORLD')
print(f8s.to_string(index=False))
