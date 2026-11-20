"""Keep the frozen ThinkTwice policy while disabling its recording camera."""
import ast
import copy
import hashlib
from pathlib import Path

from native_debug_bev import checked_logging_calls, key


class RemoveTopdown(ast.NodeTransformer):
    def __init__(self):
        self.assignments = self.entries = 0

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'topdown'):
            if not any(isinstance(x, ast.Subscript) and key(x) == 'topdown'
                       and isinstance(x.value, ast.Name) and x.value.id == 'input_data'
                       for x in ast.walk(node.value)):
                raise ValueError('Unexpected topdown conversion')
            self.assignments += 1
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        for index in reversed(range(len(node.keys))):
            name, value = node.keys[index], node.values[index]
            if isinstance(name, ast.Constant) and name.value == 'topdown':
                if not isinstance(value, ast.Name) or value.id != 'topdown':
                    raise ValueError('Unexpected topdown output')
                self.entries += 1
                del node.keys[index]
                del node.values[index]
        return self.generic_visit(node)


def without_recording_camera(module, expected_source_sha256):
    """Return a source-checked subclass, without importing or changing weights.

    The caller imports the original module with SAVE_PATH unset. Only the
    logging camera, its tick conversion/output and the unused logging folder
    assignment change. Native run_step, LiDAR history and control are inherited.
    Official sensor validation and live GPU/lifecycle checks remain mandatory.
    """
    source = Path(module.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected_source_sha256:
        raise ValueError('Frozen ThinkTwice source changed')
    if module.SAVE_PATH is not None:
        raise ValueError('Recording must be disabled before import')
    parent = getattr(module, module.get_entry_point())
    tree = ast.parse(source.read_text())
    klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == parent.__name__)
    methods = {x.name: x for x in klass.body if isinstance(x, ast.FunctionDef)}
    for name, method in methods.items():
        checked_logging_calls(method)
        if name not in ('setup', 'sensors', 'tick', 'save'):
            if any((isinstance(x, ast.Constant) and x.value == 'topdown')
                   or (isinstance(x, ast.Name) and x.id == 'topdown') for x in ast.walk(method)):
                raise ValueError('Recording camera is consumed by policy')
    setup = copy.deepcopy(methods['setup'])
    folders = [x for x in ast.walk(setup) if isinstance(x, ast.Attribute)
               and isinstance(x.value, ast.Name) and x.value.id == 'self'
               and x.attr == 'folder_name']
    if len(folders) != 1 or not isinstance(folders[0].ctx, ast.Store):
        raise ValueError('Unexpected logging folder dependency')
    for name, method in methods.items():
        if name != 'setup' and any(isinstance(x, ast.Attribute) and x.attr == 'folder_name'
                                   for x in ast.walk(method)):
            raise ValueError('Logging folder is consumed outside setup')
    assignments = [x for x in ast.walk(setup) if isinstance(x, ast.Assign)
                   and len(x.targets) == 1 and x.targets[0] is folders[0]]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Name) or assignments[0].value.id != 'string':
        raise ValueError('Unexpected recording folder assignment')
    assignments[0].value = ast.Constant(value=None)
    remover = RemoveTopdown()
    tick = remover.visit(copy.deepcopy(methods['tick']))
    if (remover.assignments, remover.entries) != (1, 1):
        raise ValueError('Expected one recording conversion and output')
    if any((isinstance(x, ast.Name) and x.id == 'topdown')
           or (isinstance(x, ast.Constant) and x.value == 'topdown') for x in ast.walk(tick)):
        raise ValueError('Unexpected remaining topdown dependency')
    namespace = dict(module.__dict__)
    executable = ast.fix_missing_locations(ast.Module(body=[setup, tick], type_ignores=[]))
    exec(compile(executable, str(source), 'exec'), namespace)

    class Adapter(parent):
        def setup(self, *args, **kwargs):
            if module.SAVE_PATH is not None:
                raise RuntimeError('Recording cannot be enabled')
            result = namespace['setup'](self, *args, **kwargs)
            if self.save_path is not None or self.folder_name is not None:
                raise RuntimeError('Native setup enabled recording')
            return result

        def sensors(self):
            sensors = parent.sensors(self)
            removed = [x for x in sensors if x['id'] == 'topdown']
            expected = dict(type='sensor.camera.rgb', x=0.0, y=0.0, z=50.0,
                roll=0.0, pitch=-90.0, yaw=0.0, width=512, height=512, fov=50.0, id='topdown')
            if removed != [expected]:
                raise ValueError('Unexpected recording camera')
            return [x for x in sensors if x['id'] != 'topdown']

        def save(self, *args, **kwargs):
            raise RuntimeError('Recording logger is disabled')

    Adapter.tick = namespace['tick']
    Adapter.__name__ = parent.__name__ + 'WithoutRecordingCamera'
    Adapter.adapter_audit = dict(native_source_sha256=digest, removed_camera='topdown',
        removed_tick_assignments=1, removed_tick_outputs=1, logging_folder_name=None,
        native_run_step_inherited=True, model_and_checkpoint_unchanged=True,
        patched_setup_ast_sha256=hashlib.sha256(ast.dump(setup).encode()).hexdigest(),
        patched_tick_ast_sha256=hashlib.sha256(ast.dump(tick).encode()).hexdigest())
    return Adapter
