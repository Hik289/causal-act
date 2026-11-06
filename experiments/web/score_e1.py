from causalact_runtime import location
# E1 scoring: 从 worker 日志提取 per-task verdict, 过滤 forum-completable 子集, 按类型 breakdown
import json, os, re, sys, glob

VWA_CFG = location('CAUSALACT_VWA_DIR', 'config_files/vwa/test_reddit')
LOGDIR = location('CAUSALACT_VWA_DIR', 'logs')

def completable_and_type():
    info = {}
    for fn in os.listdir(VWA_CFG):
        if not fn.endswith('.json'): continue
        i = fn.split('.')[0]
        if not i.isdigit(): continue
        c = json.load(open(os.path.join(VWA_CFG, fn)))
        su = c.get('start_url','')
        comp = ('9999' in su and '8888' not in su and '7770' not in su and ' |AND| ' not in su)
        it = c.get('intent','').lower()
        if 'subscrib' in it: t='subscribe'
        elif 'vote' in it: t='vote'
        elif 'navigate' in it or 'find' in it or 'comment' in it: t='navigate/find'
        elif 'post' in it or 'submit' in it or 'dm' in it or 'message' in it: t='post/submit'
        else: t='other'
        info[int(i)] = {'completable':comp, 'type':t}
    return info

def parse_verdicts(cond):
    # worker logs: [Result] (PASS) path/to/NNN.json
    v = {}
    for lf in glob.glob(f'{LOGDIR}/e1_{cond}/*.log'):
        for line in open(lf, errors='ignore'):
            m = re.search(r'\[Result\] \((PASS|FAIL)\) .*/(\d+)\.json', line)
            if m:
                v[int(m.group(2))] = 1 if m.group(1)=='PASS' else 0
    return v

def main():
    info = completable_and_type()
    comp_ids = set(i for i,d in info.items() if d['completable'])
    out = {}
    for cond in ['base','ba']:
        v = parse_verdicts(cond)
        # 只统计 completable 子集
        scored = {i:s for i,s in v.items() if i in comp_ids}
        n = len(scored); passed = sum(scored.values())
        bytype = {}
        for i,s in scored.items():
            t = info[i]['type']
            bytype.setdefault(t, [0,0])
            bytype[t][0]+=s; bytype[t][1]+=1
        out[cond] = {'n':n, 'pass':passed, 'success': passed/n if n else None,
                     'by_type': {t:{'pass':a,'n':b,'sr':a/b if b else None} for t,(a,b) in bytype.items()}}
    # paired delta on common scored ids
    vb = parse_verdicts('base'); va = parse_verdicts('ba')
    common = sorted((set(vb)&set(va)) & comp_ids)
    if common:
        bp = sum(vb[i] for i in common); ap = sum(va[i] for i in common)
        out['paired'] = {'n':len(common), 'base_pass':bp, 'ba_pass':ap,
                         'base_sr':bp/len(common), 'ba_sr':ap/len(common),
                         'delta_pp': (ap-bp)/len(common)*100}
    print(json.dumps(out, indent=2))

if __name__=='__main__': main()
