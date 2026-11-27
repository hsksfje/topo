"""Explicit execution scope. Cross-resolution is opt-in, never the default."""
import os
BASE_METHODS=['uniform','ulu','wang','cnn','meta','early']
BASE_TESTS=['test_id300','test_ood300']

def make_scope(include_cross_resolution=False):
    enabled=bool(include_cross_resolution)
    return dict(name='with_cross_resolution' if enabled else 'id_ood_only',
        include_cross_resolution=enabled,methods=BASE_METHODS+(['cgan'] if enabled else []),
        test_subsets=BASE_TESTS+(['test_cr300'] if enabled else []),
        actors=['train','test64']+(['pairs','testcr','cgan'] if enabled else []),
        training_tasks=1000,validation_tasks=100,tests_per_subset=300,OC_max_steps=500,
        planned_test_runs=(7*900 if enabled else 6*600),
        high_resolution_teachers=(1100 if enabled else 0),
        native_export_tasks_per_method=(900 if enabled else 600),
        knn='Use the separate InitSkill-KNN-v2 package; not repeated here')

def current_scope():return make_scope(os.environ.get('INITSKILL_INCLUDE_CR')=='1')

def require_grid(nx,ny):
    if max(nx,ny)>64 and not current_scope()['include_cross_resolution']:
        raise ValueError('Cross-resolution is disabled. This command cannot prepare or solve a grid above64.')
