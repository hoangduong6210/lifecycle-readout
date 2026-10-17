"""Explicit working-article assembly from the canonical wiki and public reports."""
from pathlib import Path
import json,csv,shutil,re
import argparse

parser = argparse.ArgumentParser(description="Assemble the working article from the canonical wiki and distributed aggregate records. Compilation requires a separate TeX toolchain.")
parser.add_argument("--output", type=Path, required=True, help="Destination for the mutable article source and aggregate tables")
args = parser.parse_args()
r = Path(__file__).resolve().parents[1]
s = args.output.resolve()
s.mkdir(parents=True, exist_ok=True)
canon = r / "wiki/manuscript"
def load(p):return json.loads((r/p).read_text())
def csvout(name,header,rows):
 with (s/name).open('w',newline='') as f:
  a=csv.writer(f);a.writerow(header);a.writerows(rows)
def table(caption,label,columns,head,rows,note):
 return '\n'.join([r'\begin{table}[t]',r'\caption{'+caption+r'}\label{tab:'+label+'}',r'\centering',r'\begin{tabular}{@{}'+columns+'@{}}',r'\toprule',head+r' \\',r'\midrule',*[' & '.join(map(str,row))+r' \\' for row in rows],r'\botrule',r'\end{tabular}',r'\begin{tablenotes}[flushleft]\footnotesize\item '+note+r'\end{tablenotes}',r'\end{table}'])
hard={d:load(f'results/audit/lcr-{d}-{ident}/artifacts/results.json') for d,ident in [('wikipedia','256094e9e7'),('reddit','bf7053e7a2')]}
rows=[];data=[]
for d,z in hard.items():
 c=z['per_seed_runs'][0]['coverage'];rows.append([d.capitalize(),f"{c['train']:,}",f"{c['validation']:,}",f"{c['test']:,}",f"{c['skipped_no_hard_support']:,}",f"{c['excluded_train']:,}"])
 data.append([d.capitalize(),c['total_events'],c['train'],c['validation'],c['test'],c['skipped_no_hard_support'],c['excluded_train']])
coverage=table('Conditional support under shared history-bin matching.','coverage','lrrrrr',r'Dataset & Train & Validation & Test & Unsupported & Excluded',rows,'Counts are queries or events, not candidate trajectories. Each supported query has one positive and one negative. Support counts agree across seeds. Unsupported events are omitted from supervision; training-excluded events enter neither training supervision nor training history. Totals are 157,474 Wikipedia and 672,447 Reddit events.')
csvout('Table1_coverage.csv',['Dataset','Total_events','Train_queries','Validation_queries','Test_queries','Unsupported_events','Training_excluded_events'],data)
paths=load('results/audit/scored_path_full_closure_v1/summary.json');rows=[]
for d,z in paths['datasets'].items():
 for a in z['per_seed']:
  counts={p['path']:p['violation_count'] for p in a['path_results']};rows.append([d.capitalize(),a['seed'],f"{a['candidate_trajectories']:,}",counts['implemented_flow'],f"{counts['fixed_transition']:,}",f"{counts['dynamic_transition']:,}"])
pathstable=table('Native-score path violations on supported validation candidates.','paths','llrrrr',r'Dataset & Seed & Candidates & Flow & Frozen map & Dynamic map',rows,'Entries are counts of trajectories with at least one adjacent increase above the declared threshold. Candidates count both destinations for every supported query. All six dataset/seed cohorts are retained.')
csvout('Table2_paths.csv',['Dataset','Seed','Candidates','Flow_violations','Frozen_map_violations','Dynamic_map_violations'],rows)
rep=load('results/audit/rate_repair_closure_v1/summary.json');rows=[];data=[]
for d,z in rep['datasets'].items():
 for a in z['per_seed']:
  v=a['variants'];f=v['fixed_rate_x_projected']['ap'];allr=v['all_rate_x_projected']['ap'];diff=a['primary_ap_difference']
  rows.append([d.capitalize(),a['seed'],f'{f:.7f}',f'{allr:.7f}',f'{diff*1e6:+.4f}',f"{a['all_rate_changed_candidates']:,}"])
  data.append([d.capitalize(),a['seed'],a['candidate_count'],v['gradient_unconstrained']['ap'],f,allr,diff,a['fixed_changed_candidates'],a['all_rate_changed_candidates'],a['mean_squared_distance_fixed'],a['mean_squared_distance_all_rate'],a['original_dynamic_reference_ap']])
repair=table('Same-metric no-refit repair comparison on validation queries.','repair','llrrrr',r'Dataset & Seed & Specified AP & All-rate AP & $10^6\Delta$AP & Changed',rows,r'$\Delta$AP is specified-rate minus all-rate AP; its scale is shown explicitly because differences are small. Changed counts refer to the all-rate repair; the specified-rate repair changes zero candidates in every row. The mean Reddit difference is $-1.2343\times10^{-6}$ (sample SD $1.2013\times10^{-6}$).')
csvout('Table3_repairs.csv',['Dataset','Seed','Candidates','Raw_gradient_AP','Specified_rate_AP','All_rate_AP','Specified_minus_all_AP','Specified_changed','All_changed','Specified_squared_distance','All_squared_distance','Original_dynamic_AP'],data)
labels={'base':'Neighbor affinity','base_stats':'Affinity + history','scalar_decay':'Scalar gate + history','mlp_gate':'Flexible gate + history','ordered_gate':'Ordered phase gate + history'}
rows=[];data=[]
for k,label in labels.items():
 row=[label]
 for d,z in hard.items():
  a=z['aggregate'][k]['ap'];row.append(f"${a['mean']:.5f}\\pm{a['std_ddof1']:.5f}$")
  for q in z['per_seed_runs']:
   m=q['models'][k];data.append([d.capitalize(),q['seed'],label,m['all_test']['ap'],m['all_test']['auc'],m['selected_epoch'],m['parameter_count']])
 rows.append(row)
prediction=table('Pooled test AP with common history-matched negatives.','prediction','lcc','Prediction variant & Wikipedia & Reddit',rows,'Mean and sample SD over seeds 1, 7 and 42. Test support is 977 queries per Wikipedia seed and 11,002 per Reddit seed. These are conditional internal-architecture comparisons; the values are not comparable with easier-negative benchmark tables.')
csvout('Table4_prediction.csv',['Dataset','Seed','Variant','Test_AP','Test_AUC','Selected_epoch','Parameters'],data)
lf=load('results/audit/lastfm_model_closure_v1/summary.json');ll={'history_only':'History only','dose_mlp':'Flexible time/dose residual','scalar_clock':'Proportional-rate clock','phase_competition':'Independent-rate phase','shuffled_phase':'Alternate-reference phase','activity_mlp':'Activity-only residual'};rows=[]
for k,label in ll.items():
 a=lf['arms'][k];ap=a['metrics']['ap'];rows.append([label,a['branch_parameter_count'],f"${ap['mean']:.6f}\\pm{ap['sample_sd']:.6f}$"])
lastfm=table('January Last.fm development comparison.','lastfm','lrc','Prediction variant & Branch parameters & AP',rows,'Mean and sample SD over seeds 101, 202 and 303; 40,600 validation queries per seed. Every arm uses the same 1,054,378-parameter history backbone. Phase versus flexible residual is the primary same-input, equal-branch-parameter contrast. These are checkpoint-selection-window outcomes.')
strata=load('results/audit/lastfm_strata_v1/results.json');feb=load('results/audit/lastfm_february_audit_v1/results.json');data=[];contrasts=[]
cl={'primary_shape':'Phase minus flexible','phase_specific_rates':'Phase minus proportional clock','profile_alignment':'Phase minus alternate reference','profile_information':'Flexible minus activity only','residual_vs_history':'Phase minus history only'}
for month,z in [('January',strata),('February',feb)]:
 for a in z['per_seed']:
  for cohort,c in a['cohorts'].items():
   n=c.get('query_count',c.get('queries'));u=c.get('user_count',c.get('users'))
   for k,m in c['models'].items():data.append([month,a['seed'],cohort,ll[k],n,u,m['ap'],m['auc'],m['pair_accuracy']])
   for k,label in cl.items():
    pairs={'primary_shape':('phase_competition','dose_mlp'),'phase_specific_rates':('phase_competition','scalar_clock'),'profile_alignment':('phase_competition','shuffled_phase'),'profile_information':('dose_mlp','activity_mlp'),'residual_vs_history':('phase_competition','history_only')}
    t,ctrl=pairs[k];contrasts.append([month,a['seed'],cohort,label,n,u,c['models'][t]['ap']-c['models'][ctrl]['ap']])
csvout('LastFM_all_cohorts.csv',['Month','Seed','Cohort','Variant','Queries','Users','AP','AUC','Pair_accuracy'],data)
csvout('LastFM_all_contrasts.csv',['Month','Seed','Cohort','Contrast','Queries','Users','AP_difference'],contrasts)
front=(canon/'Article-Front-Matter.tex').read_text();body=(canon/'Article-Body.tex').read_text()
for k,v in {'COVERAGE':coverage,'PATHS':pathstable,'REPAIR':repair,'PREDICTION':prediction,'LASTFM':lastfm}.items():
 assert body.count('@@'+k+'@@')==1;body=body.replace('@@'+k+'@@',v)
assert '@@' not in body
(s/'main.tex').write_text(front+'\n'+body+'\n\\bibliography{references}\n\\end{document}\n')
shutil.copy2(canon/'Article-References.bib',s/'references.bib')
template = r / 'paper/working/article'
for name in ('sn-jnl.cls', 'sn-vancouver-num.bst'):
 if (template / name).resolve() != (s / name).resolve(): shutil.copyfile(template / name, s / name)
for figure in template.glob('Fig[1-8].pdf'):
 if figure.resolve() != (s / figure.name).resolve(): shutil.copyfile(figure, s / figure.name)
abstract=re.search(r'\\abstract\{(.*?)\}\n\\keywords',front,re.S).group(1);print('Abstract words:',len(re.sub(r'\\textbf\{([^}]+)\}',r'\1',abstract).split()));print('Tables generated from original records; 576 cohort/model rows and 480 paired contrasts expected:',len(data),len(contrasts))
