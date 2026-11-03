"""Causal six-frame Radar, full target alignment before masked 4x4 maxima.

Raw official inputs are read-only. The schema contains no target or fitted norm.
"""
from pathlib import Path
import hashlib, json, os, time
import numpy as np
from precip_forecast.realdata.pseudotime import PseudoTime,add_hours,extract_time_token
from precip_forecast.realdata.task51_radar import _nearest_indices

SCHEMA='task65-radar-sixframe-block4-v1'
SLOTS=((-2,54),(-1,6),(-1,18),(-1,30),(-1,42),(-1,54))
OFFSETS=(-66,-54,-42,-30,-18,-6)
FIELDS=('zmax','midmax')

def hash_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def slots(anchor):
    return [(add_hours(anchor,h),m) for h,m in SLOTS]

def block_max(value,valid,block=4):
    if value.shape!=valid.shape or value.ndim<2 or any(n%block for n in value.shape[-2:]):raise ValueError('Exact block shape')
    if valid.dtype!=np.bool_:raise ValueError('Independent boolean mask')
    shape=(*value.shape[:-2],value.shape[-2]//block,block,value.shape[-1]//block,block)
    v=valid.reshape(shape);out_valid=v.any(axis=(-3,-1))
    out=np.where(valid,value,-np.inf).reshape(shape).max(axis=(-3,-1))
    return np.where(out_valid,out,0).astype(np.float32),out_valid

def vertical_fields(raw,levels):
    if raw.ndim!=3 or len(levels)!=raw.shape[0]:raise ValueError('Vertical shape')
    middle=(levels>=3)&(levels<=7)
    if not middle.any():raise ValueError('No audited 3..7 km levels: unresolved schema')
    a=np.asarray(np.ma.asarray(raw).astype(np.float32).filled(np.nan),np.float32)
    mask=np.isfinite(a)&(a>-900)
    a=np.clip(np.where(mask,a,0),-20,70)
    values=[];masks=[]
    for choose in (np.ones(len(levels),bool),middle):
        m=mask[choose].any(0);v=np.where(mask[choose],a[choose],-np.inf).max(0)
        values.append(np.where(m,v,0));masks.append(m)
    return np.asarray(values,np.float32),np.asarray(masks,bool)

def snapshot_quality(valid,static_domain,offsets=OFFSETS):
    if valid.shape!=(6,2,128,192) or static_domain.shape!=(512,768):raise ValueError('Quality scope')
    if static_domain.dtype!=np.bool_:raise ValueError('Static domain boolean')
    covered=static_domain.reshape(128,4,192,4).any((1,3))
    if not covered.any():raise ValueError('Empty static domain')
    fractions=valid[:,0,covered].mean(-1)
    usable=fractions>=.10
    latest_age=float(-np.asarray(offsets)[usable].max()) if usable.any() else None
    return dict(fractions=fractions.tolist(),snapshots_valid=usable.tolist(),count=int(usable.sum()),latest_age_minutes=latest_age,
        available=bool(usable.sum()>=2 and latest_age<=18),static_block_denominator=int(covered.sum()))

def to_model(values,valid):
    if values.shape!=(6,2,128,192) or valid.shape!=values.shape:raise ValueError('Sequence shape')
    values=np.asarray(values,np.float32);valid=np.asarray(valid,bool)
    if not np.isfinite(values).all():raise ValueError('Nonfinite cache must not become fallback')
    mapped=np.where(valid,(np.clip(values,-20,70)+20)/90,0)
    return np.concatenate((mapped,valid.astype(np.float32)),axis=1).astype(np.float32)

def checked_header(ds,path,target_lat,target_lon):
    from netCDF4 import num2date
    if not {'REF','time','level','latitude','longitude'}.issubset(ds.variables):raise ValueError('Radar required schema')
    ref=ds.variables['REF'];ti=ds.variables['time'];lv=ds.variables['level']
    if ref.dimensions!=('time','level','latitude','longitude') or getattr(ref,'units',None)!='dBZ':raise ValueError('REF dimensions or units')
    if getattr(lv,'units',None)!='km':raise ValueError('Unresolved height unit; no conversion guess')
    times=num2date(np.asarray(ti[:],np.float64),units=ti.units,calendar=getattr(ti,'calendar','standard'),only_use_cftime_datetimes=True)
    token=extract_time_token(path);t=PseudoTime.parse(token)
    expected=(t.year,t.month,t.day,t.hour)
    if len(times)!=10 or [(x.year,x.month,x.day,x.hour) for x in times]!=[expected]*10 or [x.minute for x in times]!=list(range(0,60,6)) or any(x.second for x in times):
        raise ValueError('Filename/internal Radar time conflict')
    lat=np.asarray(ds.variables['latitude'][:],np.float64);lon=np.asarray(ds.variables['longitude'][:],np.float64)
    for a in (lat,lon,target_lat,target_lon):
        if not np.isfinite(a).all() or not (np.all(np.diff(a)>0) or np.all(np.diff(a)<0)):raise ValueError('Non-monotonic coordinate')
    yi=_nearest_indices(lat,target_lat);xi=_nearest_indices(lon,target_lon)
    if yi.shape!=(512,) or xi.shape!=(768,) or not np.all(np.abs(np.diff(yi))==1) or not np.all(np.abs(np.diff(xi))==1):raise ValueError('Full target coordinate rectangle not contiguous')
    levels=np.asarray(lv[:],np.float32)
    if not ((levels>=3)&(levels<=7)).any():raise ValueError('No audited MIDMAX height band')
    identity=dict(schema=SCHEMA,source_lat_sha=hash_array(lat),source_lon_sha=hash_array(lon),target_lat_sha=hash_array(target_lat),target_lon_sha=hash_array(target_lon),
        y_indices=yi.tolist(),x_indices=xi.tolist(),block=4,block_anchor='full_target[4*i,4*j]; no center shift',levels_km=levels.tolist(),
        level_unit='km',value_unit='dBZ',decoded_times=[str(x) for x in times],token=token,source_shape=list(ref.shape),source_chunks=ref.chunking())
    return yi,xi,levels,identity

def extract_hour(task):
    """One spawn worker; at most one time-frame/all-level target rectangle in RAM."""
    from netCDF4 import Dataset
    start=time.monotonic();path=Path(task['path']);minutes=task['minutes'];cache=Path(task['cache'])
    if any(m not in (6,18,30,42,54) for m in minutes):raise ValueError('Only predeclared source frame minutes')
    target_lat=np.load(task['target_lat'],allow_pickle=False).astype(np.float64)
    target_lon=np.load(task['target_lon'],allow_pickle=False).astype(np.float64)
    output=[]
    with Dataset(path,'r') as ds:
        yi,xi,levels,header=checked_header(ds,path,target_lat,target_lon)
        y0,y1=int(yi.min()),int(yi.max())+1;x0,x1=int(xi.min()),int(xi.max())+1
        for minute in minutes:
            tick=time.monotonic();frame_index=minute//6
            raw=ds.variables['REF'][frame_index,:,y0:y1,x0:x1]
            raw=np.ma.take(np.ma.take(raw,yi-y0,axis=1),xi-x0,axis=2)
            full,full_valid=vertical_fields(raw,levels);values,valid=block_max(full,full_valid)
            # Direct alternative reduction on selected native voxels, including boundary blocks.
            manual=[]
            for by,bx in ((0,0),(31,47),(64,96),(127,191)):
                for c,choose in enumerate((np.ones(len(levels),bool),(levels>=3)&(levels<=7))):
                    sub=np.asarray(np.ma.asarray(raw[choose,by*4:by*4+4,bx*4:bx*4+4]).astype(np.float32).filled(np.nan))
                    good=np.isfinite(sub)&(sub>-900);expected=float(np.clip(sub[good],-20,70).max()) if good.any() else 0.
                    if bool(valid[c,by,bx])!=bool(good.any()) or float(values[c,by,bx])!=expected:raise RuntimeError('Independent voxel/block check failed')
                    manual.append(dict(block=[by,bx],field=FIELDS[c],valid=bool(good.any()),difference=0.))
            half=values.astype(np.float16);error=float(np.abs(half.astype(np.float32)-values)[valid].max()) if valid.any() else 0.
            if error>.05:raise RuntimeError('float16 error exceeds .05dBZ')
            coord_id=hashlib.sha256(json.dumps({k:v for k,v in header.items() if k not in ('decoded_times','token')},sort_keys=True).encode()).hexdigest()
            key=f"{task['process']}|{header['token']}|{minute:02d}"
            identity=dict(key=key,schema=SCHEMA,coordinate_transform_sha256=coord_id,source_path=str(path),source_size=path.stat().st_size,source_mtime_ns=path.stat().st_mtime_ns,minute=minute,
                source_hash_policy='metadata + REAL_DATA_EPOCH; not a full source content SHA',values_sha256=hash_array(half),valid_sha256=hash_array(valid.astype(np.uint8)))
            destination=cache/task['process']/(header['token']+f'_{minute:02d}.npz');receipt=destination.with_suffix('.json')
            destination.parent.mkdir(parents=True,exist_ok=True)
            if receipt.exists():
                previous=json.loads(receipt.read_text())
                if previous['identity']!=identity or not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest()!=previous['sha256']:raise RuntimeError('Existing frame differs; do not overwrite')
            else:
                if destination.exists():raise RuntimeError('Unreceipted existing complete cache file: reconcile before reuse')
                tmp=destination.with_name(destination.name+'.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,values=half,valid=valid.astype(np.uint8));f.flush();os.fsync(f.fileno())
                tmp.replace(destination)
                previous=dict(identity=identity,path=str(destination),bytes=destination.stat().st_size,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),float16_error_dbz_max=error,
                    valid_fraction=valid.mean(axis=(1,2)).tolist(),manual_checks=manual,read_elapsed_seconds=time.monotonic()-tick)
                rt=receipt.with_name(receipt.name+'.tmp');rt.write_text(json.dumps(previous,sort_keys=True,indent=2)+'\n');rt.replace(receipt)
            output.append(previous)
            del raw,full,full_valid,values,valid,half
    return dict(path=str(path),source_bytes=path.stat().st_size,minutes=minutes,frames=output,header=header,worker_pid=os.getpid(),wall_seconds=time.monotonic()-start)
