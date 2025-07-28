import six
import inspect


def is_str(x):
    """Whether the input is an string instance."""
    return isinstance(x, six.string_types)


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f"(name={self._name}, "
        format_str += f"items={list(self._module_dict.keys())})"
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError(f"module must be a class, "
                            f"but got {type(module_class)}")
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f"{module_name} already registered in {self.name}")
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and "type" in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop("type")
    if is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, "
                        f"but got {type(obj_type)}")
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)

# def build_from_cfg(cfg, registry, default_args=None):
#     """Build a module from config dict.

#     Args:
#         cfg (dict): Config dict. It should at least contain the key "type".
#         registry (:obj:`Registry`): The registry to search the type from.
#         default_args (dict, optional): Default initialization arguments.

#     Returns:
#         obj: The constructed object.
#     """
#     assert isinstance(cfg, dict) and "type" in cfg
#     assert isinstance(default_args, dict) or default_args is None
#     args = cfg.copy()
#     obj_type = args.pop("type")
#     if is_str(obj_type):
#         obj_cls = registry.get(obj_type)
#         if obj_cls is None:
#             raise KeyError(f"{obj_type} not in the {registry.name} registry")
#     elif inspect.isclass(obj_type):
#         obj_cls = obj_type
#     else:
#         raise TypeError(f"type must be a str or valid type, "
#                         f"but got {type(obj_type)}")
#     if default_args is not None and obj_type == "CSPDarknet":
#         for name, value in default_args.items():
#             if name != 'cfg':  # Remove 'cfg' from default_args
#                 args.setdefault(name, value)
#     return obj_cls(**args)