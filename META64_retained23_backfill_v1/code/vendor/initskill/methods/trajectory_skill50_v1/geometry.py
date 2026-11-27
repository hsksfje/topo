"""Task-derived geometry; supports every load and the original constrained DOFs."""
import copy
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from initskill.io import digest
from initskill.initializers import calibrate
from initskill.sensitivity_initializers import hole_field, scaled_volume

METHODS = ('U', 'H4_B010', 'H4_B020', 'H4_B035', 'H9_B010',
           'PG1', 'PG2', 'PG3', 'PG4', 'PG5', 'PG6', 'R4_B020', 'R9_B020',
           'ENERGY_LOG', 'ENERGY_SQRT', 'MIX_PG1_H4', 'MIX_PG2_H9')
WIDTHS = (.06, .10, .16, .22, .12, .18)
BETAS = (.2, .35, .5, .65, .8, 1.)
FEATURES = ('volume_fraction', 'aspect_ratio', 'load_y', 'horizontal_share', 'loaded_nodes', 'support_components')


def context(task):
    nx, ny = task['elements_xy']
    loads, supports = {}, {}
    def point(node):
        col, row = divmod(node, ny+1)
        return dict(u=col/nx, v=1-row/ny)
    for dof, value in task['forces_sparse']:
        node, axis = divmod(int(dof), 2)
        p = loads.setdefault(node, dict(node=node, **point(node), fx=0., fy=0.))
        p['fx' if axis == 0 else 'fy'] += float(value)
    for dof in task['fixed_dofs']:
        node, axis = divmod(int(dof), 2)
        p = supports.setdefault(node, dict(node=node, **point(node), fixed_x=False, fixed_y=False))
        p['fixed_x' if axis == 0 else 'fixed_y'] = True
    return dict(elements_xy=[nx,ny], volume_fraction=task['volume_fraction'],
                loads=list(loads.values()), supports=list(supports.values()),
                coordinates='u=x/W,v=y/H; y upwards; width/radius divided by H')


def support_groups(task):
    """4-connected constrained nodes, separated by constrained-direction signature."""
    c = context(task)
    ny = task['elements_xy'][1]
    points = {p['node']:p for p in c['supports']}
    todo, groups = set(points), []
    while todo:
        node = min(todo)
        todo.remove(node)
        component, queue = [node], [node]
        signature = (points[node]['fixed_x'], points[node]['fixed_y'])
        while queue:
            n = queue.pop()
            col, row = divmod(n, ny+1)
            for other in (n-1, n+1, n-(ny+1), n+(ny+1)):
                if other not in todo:
                    continue
                oc, rr = divmod(other, ny+1)
                if abs(oc-col)+abs(rr-row) != 1:
                    continue
                p = points[other]
                if (p['fixed_x'], p['fixed_y']) == signature:
                    todo.remove(other)
                    queue.append(other)
                    component.append(other)
        groups.append([points[n] for n in sorted(component)])
    return groups


def features(task):
    c = context(task)
    weights = [abs(p['fx'])+abs(p['fy']) for p in c['loads']]
    total = sum(weights)
    if total <= 0:
        raise ValueError('Nonzero load required')
    return dict(volume_fraction=float(task['volume_fraction']),
        aspect_ratio=task['elements_xy'][0]/task['elements_xy'][1],
        load_y=sum(w*p['v'] for w,p in zip(weights,c['loads']))/total,
        horizontal_share=sum(abs(p['fx']) for p in c['loads'])/total,
        loaded_nodes=float(len(c['loads'])), support_components=float(len(support_groups(task))))


def pg_paths(task, template, width_scale=1., bend=0.):
    """A/B/M per constrained component; all actual loaded nodes participate."""
    c = context(task)
    groups = support_groups(task)
    routes = []
    for group in groups:
        # Extreme points on the component's longer physical axis; center is a real support.
        xy = np.array([[p['u']*task['elements_xy'][0], p['v']*task['elements_xy'][1]] for p in group])
        axis = int(np.argmax(np.ptp(xy, axis=0)))
        a, b = group[int(np.argmin(xy[:,axis]))], group[int(np.argmax(xy[:,axis]))]
        m = group[int(np.argmin(np.sum((xy-xy.mean(axis=0))**2,axis=1)))]
        starts = ([a,b], [m], [a,b,m])[(template-1)%3]
        for load in c['loads']:
            for start in starts:
                p, q = [start['u'],start['v']], [load['u'],load['v']]
                if p == q:
                    continue
                points = [p,q]
                if bend:
                    points = [p,[(p[0]+q[0])/2, float(np.clip((p[1]+q[1])/2+bend,0,1))],q]
                path = dict(width_H=WIDTHS[template-1]*width_scale, points=points)
                if path not in routes:
                    routes.append(path)
    if not routes:
        raise ValueError('No nonzero support-to-load path')
    return routes


def regular_holes(count):
    side = int(math.sqrt(count))
    return [dict(cx=float(x),cy=float(y),radius_H=.25/side)
            for x in (np.arange(side)+.5)/side for y in (np.arange(side)+.5)/side]


def teacher_pool(task):
    out = {'U':dict(tool='uniform')}
    for beta in (.1,.2,.35):
        out[f'H4_B{int(beta*100):03d}'] = dict(tool='holes',beta=beta,holes=regular_holes(4))
    out['H9_B010'] = dict(tool='holes',beta=.1,holes=regular_holes(9))
    for i in range(1,7):
        out[f'PG{i}'] = dict(tool='pg',template=i,width_scale=1.,bend=0.,beta=BETAS[i-1])
    w,h = task['elements_xy']
    for count in (4,9):
        seed = int(digest([20261112,task['task_id'],'random_holes',count])[:16],16)
        rng = np.random.default_rng(seed)
        # Scale random circles to the short side so portrait grids remain packable.
        radius, placed = .25*min(w,h)/math.sqrt(count), []
        for _ in range(20000):
            r = float(radius*rng.uniform(.75,1.25))
            x,y = float(rng.uniform(r,w-r)),float(rng.uniform(r,h-r))
            if all(np.hypot(x-a,y-b)>1.05*(r+c) for a,b,c in placed):
                placed.append((x,y,r))
            if len(placed)==count:
                break
        if len(placed)!=count:
            raise ValueError('Random packing exhausted')
        out[f'R{count}_B020'] = dict(tool='holes',beta=.2,
            holes=[dict(cx=x/w,cy=y/h,radius_H=r/h) for x,y,r in placed])
    out['ENERGY_LOG'] = dict(tool='energy_log')
    out['ENERGY_SQRT'] = dict(tool='energy_sqrt')
    for i,count in ((1,4),(2,9)):
        out[f'MIX_PG{i}_H{count}'] = dict(tool='hybrid',template=i,width_scale=1.,bend=0.,beta=.2,holes=regular_holes(count))
    if tuple(out)!=METHODS:
        raise AssertionError('Frozen 17-slot order')
    return out


def number(value, lo, hi):
    if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:
        raise ValueError(f'Expected finite number in [{lo},{hi}]')


def validate_recipe(recipe):
    if not isinstance(recipe,dict):
        raise ValueError('Recipe object required')
    tool = recipe.get('tool')
    fields = {'uniform':{'tool'},'energy_log':{'tool'},'energy_sqrt':{'tool'},
              'holes':{'tool','beta','holes'},'pg':{'tool','template','width_scale','bend','beta'},
              'hybrid':{'tool','template','width_scale','bend','beta','holes'},
              'free_skeleton':{'tool','beta','nodes','edges'},
              'free_hybrid':{'tool','beta','nodes','edges','holes'}}
    if tool not in fields or set(recipe)!=fields[tool]:
        raise ValueError('Recipe tool/fields')
    if 'beta' in recipe:
        number(recipe['beta'],0,1)
    if tool in ('pg','hybrid'):
        if type(recipe['template']) is not int or not 1<=recipe['template']<=6:
            raise ValueError('Template must be integer 1..6')
        number(recipe['width_scale'],.5,1.5)
        number(recipe['bend'],-.2,.2)
    if tool in ('free_skeleton','free_hybrid'):
        nodes,edges=recipe['nodes'],recipe['edges']
        if not isinstance(nodes,list) or not 2<=len(nodes)<=32:
            raise ValueError('Free skeleton requires2..32 nodes')
        for node in nodes:
            if not isinstance(node,dict) or set(node)!={'u','v'}:
                raise ValueError('Node fields u/v')
            number(node['u'],0,1);number(node['v'],0,1)
        if len({(n['u'],n['v']) for n in nodes})!=len(nodes):
            raise ValueError('Duplicate graph nodes')
        if not isinstance(edges,list) or not 1<=len(edges)<=64:
            raise ValueError('Free skeleton requires1..64 edges')
        seen,used=set(),set()
        for edge in edges:
            if not isinstance(edge,dict) or set(edge)!={'start','end','width_H'}:
                raise ValueError('Edge fields start/end/width_H')
            a,b=edge['start'],edge['end']
            if any(type(i) is not int or not 0<=i<len(nodes) for i in (a,b)) or a==b:
                raise ValueError('Invalid edge node indices')
            key=tuple(sorted((a,b)))
            if key in seen:
                raise ValueError('Duplicate undirected edge')
            seen.add(key);used.update((a,b))
            number(edge['width_H'],.01,.4)
        if used!=set(range(len(nodes))):
            raise ValueError('Every graph node must participate in an edge')
    if 'holes' in recipe:
        if not isinstance(recipe['holes'],list) or not 1<=len(recipe['holes'])<=24:
            raise ValueError('1..24 holes required')
        for hole in recipe['holes']:
            if not isinstance(hole,dict) or set(hole)!={'cx','cy','radius_H'}:
                raise ValueError('Hole fields')
            number(hole['cx'],0,1); number(hole['cy'],0,1); number(hole['radius_H'],.005,.5)
    return copy.deepcopy(recipe)


def path_field(model, paths):
    best = np.full(len(model.cells),-3.)
    for path in paths:
        distance = np.full(len(model.cells),np.inf)
        points = np.array(path['points'])*[model.nx,model.ny]
        for a,b in zip(points[:-1],points[1:]):
            delta = b-a
            norm = float(delta@delta)
            if norm<=0:
                raise ValueError('Zero length segment')
            t = np.clip((model.centers-a)@delta/norm,0,1)
            distance = np.minimum(distance,np.linalg.norm(model.centers-a-t[:,None]*delta,axis=1))
        best = np.maximum(best,np.clip((path['width_H']*model.ny/2-distance)/(.04*model.ny),-3,3))
    return best


def decode(model, task, recipe):
    p = validate_recipe(recipe)
    tool = p['tool']
    metadata = dict(recipe=p,extra_fe=0)
    if tool=='uniform':
        return model.initial(), metadata
    if tool in ('energy_log','energy_sqrt'):
        e = model.evaluate(model.initial())
        metadata.update(extra_fe=1,reference_compliance=e['compliance'])
        if tool=='energy_log':
            field = np.log(np.maximum(e['energy'],1e-30))
            field = (field-np.median(field))/max(float(np.std(field)),1e-12)
            return calibrate(model,field,.8), metadata
        energy = gaussian_filter(e['energy'].reshape(model.ny,model.nx,order='F'),sigma=model.config.rmin,mode='nearest').ravel(order='F')
        energy = np.maximum(energy,1e-30)
        return scaled_volume(model,np.sqrt(energy/energy.mean())), metadata
    field = None
    if tool in ('holes','hybrid','free_hybrid'):
        items = [(h['cx']*model.nx,h['cy']*model.ny,h['radius_H']*model.ny) for h in p['holes']]
        field = hole_field(model,items)
    if tool in ('pg','hybrid','free_skeleton','free_hybrid'):
        if tool in ('free_skeleton','free_hybrid'):
            paths=[dict(width_H=e['width_H'],points=[[p['nodes'][i]['u'],p['nodes'][i]['v']]
                    for i in (e['start'],e['end'])]) for e in p['edges']]
        else:
            paths = pg_paths(task,p['template'],p['width_scale'],p['bend'])
        metadata['realized_paths'] = paths
        skeleton = path_field(model,paths)
        field = skeleton if field is None else field+skeleton
    layout = calibrate(model,field)
    return (1-p['beta'])*model.initial()+p['beta']*layout, metadata
