from types import FunctionType
from fewlabel import training as original
from .models import build_model


def with_model_factory(factory):
    namespace = dict(original.train_job.__globals__)
    namespace["build_model"] = factory
    function = FunctionType(original.train_job.__code__, namespace,
                            original.train_job.__name__, original.train_job.__defaults__,
                            original.train_job.__closure__)
    function.__kwdefaults__ = original.train_job.__kwdefaults__
    function.__module__ = __name__
    return function


train_job = with_model_factory(build_model)
