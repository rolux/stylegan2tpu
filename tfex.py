import tensorflow as tf
from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

_devices = None
_has_gpu = False

def has_gpu():
  global _devices
  global _has_gpu
  if _devices is None:
    _devices = tf.get_default_session().list_devices()
    _has_gpu = len([x.name for x in _devices if ':GPU' in x.name]) > 0
  return _has_gpu

def device(name=''):
  if name is None:
    return tf.device(None)
  if 'gpu' in name:
    if has_gpu():
      return tf.device(name)
  if 'cpu' in name:
    return tf.device(name)
  return nullcontext()