import sys,re,collections
S=collections.OrderedDict()
for l in open(sys.argv[1]):
    m=re.search(r'MOMENTUM mode=(\S+) np=(\d+) thr=(\d+)',l)
    if not m: continue
    k=(m.group(1),int(m.group(2)))
    d=S.setdefault(k,dict(n=0,dP=0,dX=0,dXpos=0,dLvel=0,ovl=0,conf=0,gap=0,res=0,clamps=0,ctl=set(),fail=0))
    d['n']+=1
    for f in ('dP','dX','dXpos','dLvel','ovl'):
        mm=re.search(r' '+f+r'=([0-9.e+-]+)',l)
        if mm: d[f]=max(d[f],float(mm.group(1)))
    mm=re.search(r'CONFLICTS[^M]* vel=(\d+) pos=(\d+)',l)
    if mm: d['conf']=max(d['conf'],int(mm.group(1)),int(mm.group(2)))
    mm=re.search(r'maxGap/delta=(\S+) residual/R=(\S+)',l)
    if mm: d['gap']=max(d['gap'],float(mm.group(1))); d['res']=max(d['res'],float(mm.group(2)))
    mm=re.search(r'orphanClamps=(\d+)',l)
    if mm: d['clamps']=max(d['clamps'],int(mm.group(1)))
    mm=re.search(r'mlLevels=(\d+) mlHubAggregated=(\d+)',l)
    if mm: d['ctl'].add(f"L{mm.group(1)}A{mm.group(2)}")
    if 'FAILED' in l: d['fail']+=1
for (m,np_),d in S.items():
    x=f"{m:18s} np{np_} n={d['n']} dP={d['dP']:.2e} dX={d['dX']:.2e} dXpos={d['dXpos']:.2e} dLvel={d['dLvel']:.2e} ovl={d['ovl']:.3e} conf={d['conf']}"
    if d['gap'] or d['res']: x+=f" gap/d={d['gap']:.3f} res/R={d['res']:.1e}"
    if m=='cluster_poisson': x+=f" clamps={d['clamps']}"
    if d['ctl']: x+=' ctl='+','.join(sorted(d['ctl']))
    x+=f" FAIL={d['fail']}" if d['fail'] else ''
    print(x)
