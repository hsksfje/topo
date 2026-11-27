"""CPU scheduling only. Scope and physics are immutable separately."""
import os

def cpu_allocation(cpus=28,ram_gb=64):
    import psutil
    from initskill.methods.benchmark_init_v1.scope import current_scope
    cpus=int(cpus);requested=float(ram_gb)
    if cpus<3 or requested<8:raise ValueError('At least3 CPU slots and8GiB RAM required')
    actual=psutil.virtual_memory().total/1024**3;ram=min(requested,actual)
    reserve=min(16.,max(8.,ram/4))
    cross=current_scope()['include_cross_resolution']
    if cross:
        small=2;large=min(cpus-4,max(1,int((ram-reserve-small*.5)//2)))
    else:small=min(cpus-2,max(1,int((ram-reserve)/.5)));large=0
    return dict(cpu_budget=cpus,requested_ram_gib=requested,detected_ram_gib=actual,
        planning_ram_gib=ram,reserved_ram_gib=reserve,small_workers=small,
        cr_workers=large,teacher_workers=large,large_slots=large,
        reserved_training_cpus=2,maximum_optimization_tasks=small+large,
        note='One training stage at a time. No large-grid slots exist in the default ID/OOD scope.')

def settings(profile):
    h=profile.get('hardware',dict(cpus=28,ram_gb=64))
    return cpu_allocation(h['cpus'],h['ram_gb'])
