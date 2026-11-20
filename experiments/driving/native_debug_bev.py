"""Exclude a verified logging-only BEV camera from a frozen native agent."""
import ast
import copy
import hashlib
from pathlib import Path


def key(node):
    # Python 3.8 wraps Subscript.slice in Index.
    value = node.slice
    if isinstance(value, ast.Index):
        value = value.value
    return value.value if isinstance(value, ast.Constant) else None


def bev_read(node, variable):
    return (isinstance(node, ast.Subscript) and key(node) == 'bev'
            and isinstance(node.value, ast.Name) and node.value.id == variable)


def requires_logging(node):
    if isinstance(node, ast.Compare):
        return (isinstance(node.left, ast.Name) and node.left.id == 'SAVE_PATH'
                and len(node.ops) == 1 and isinstance(node.ops[0], ast.IsNot)
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value is None)
    if isinstance(node, ast.BoolOp):
        checks = [requires_logging(x) for x in node.values]
        return any(checks) if isinstance(node.op, ast.And) else all(checks)
    return False


def checked_logging_calls(node, guarded=False):
    if isinstance(node, ast.If):
        checked_logging_calls(node.test, guarded)
        for child in node.body:
            checked_logging_calls(child, guarded or requires_logging(node.test))
        for child in node.orelse:
            checked_logging_calls(child, guarded)
        return
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'
            and node.func.attr == 'save' and not guarded):
        raise ValueError('Native logging call is not disabled by SAVE_PATH=None')
    for child in ast.iter_child_nodes(node):
        checked_logging_calls(child, guarded)


class RemoveTickDebug(ast.NodeTransformer):
    def __init__(self):
        self.assignments = self.entries = 0

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'bev'):
            if not any(bev_read(x, 'input_data') for x in ast.walk(node.value)):
                raise ValueError('Unexpected BEV assignment; manual audit required')
            self.assignments += 1
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        for index in reversed(range(len(node.keys))):
            entry = node.keys[index]
            if isinstance(entry, ast.Constant) and entry.value == 'bev':
                value = node.values[index]
                if not isinstance(value, ast.Name) or value.id != 'bev':
                    raise ValueError('Unexpected BEV output; manual audit required')
                self.entries += 1
                del node.keys[index]
                del node.values[index]
        return self.generic_visit(node)


def without_debug_bev(module, expected_source_sha256):
    """Return an isolated subclass; never rewrite the native source or model.

    Only the logging camera, its conversion and its returned logging field
    are removed. Reject sources that use it elsewhere or can enable logging.
    The caller still runs the official sensor validator on the resulting list.
    """
    source = Path(module.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected_source_sha256:
        raise ValueError('Frozen native agent source changed')
    if module.SAVE_PATH is not None:
        raise ValueError('Debug logging must be disabled before importing the adapter')
    parent = getattr(module, module.get_entry_point())
    tree = ast.parse(source.read_text())
    klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == parent.__name__)
    tick = next(x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name == 'tick')
    for method in klass.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        if method.name not in ('tick', 'save', 'setup', 'sensors'):
            if any(isinstance(x, ast.Constant) and x.value == 'bev' for x in ast.walk(method)):
                raise ValueError('BEV camera is consumed outside the disabled logger')
        checked_logging_calls(method)
    remover = RemoveTickDebug()
    patched = remover.visit(copy.deepcopy(tick))
    if (remover.assignments, remover.entries) != (1, 1):
        raise ValueError('Expected exactly one logging conversion and output')
    if any((isinstance(x, ast.Name) and x.id == 'bev')
           or (isinstance(x, ast.Constant) and x.value == 'bev') for x in ast.walk(patched)):
        raise ValueError('Unexpected remaining BEV dependency')
    namespace = dict(module.__dict__)
    executable = ast.fix_missing_locations(ast.Module(body=[patched], type_ignores=[]))
    exec(compile(executable, str(source), 'exec'), namespace)

    class Adapter(parent):
        def setup(self, *args, **kwargs):
            if module.SAVE_PATH is not None:
                raise RuntimeError('Debug logging cannot be enabled')
            result = super().setup(*args, **kwargs)
            if self.save_path is not None:
                raise RuntimeError('Native setup enabled debug logging')
            return result

        def sensors(self):
            sensors = parent.sensors(self)
            removed = [x for x in sensors if x['id'] == 'bev']
            expected = dict(type='sensor.camera.rgb', x=0.0, y=0.0, z=50.0,
                            roll=0.0, pitch=-90.0, yaw=0.0, width=512, height=512, fov=50.0, id='bev')
            if removed != [expected]:
                raise ValueError('Unexpected logging camera; manual audit required')
            return [x for x in sensors if x['id'] != 'bev']

        def save(self, *args, **kwargs):
            raise RuntimeError('BEV debug logger is disabled')

    Adapter.tick = namespace['tick']
    Adapter.__name__ = parent.__name__ + 'WithoutDebugBEV'
    Adapter.adapter_audit = dict(native_source_sha256=digest,
        removed_camera='bev', removed_tick_assignments=1, removed_tick_outputs=1,
        native_run_step_inherited=True, model_and_checkpoint_unchanged=True,
        patched_tick_ast_sha256=hashlib.sha256(ast.dump(patched).encode()).hexdigest())
    return Adapter
