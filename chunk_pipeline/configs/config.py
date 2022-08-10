# adapted from https://github.com/pallets/flask/blob/36af821edf741562cdcb6c60d63f23fa9a1d8776/src/flask/config.py#L1
import os
import types


class Config(dict):
    def __init__(self, root_path):
        super().__init__({})
        self.root_path = root_path

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if item is None:
            raise KeyError(key)
        return item

    def from_pyfile(self, filename):
        filename = os.path.join(self.root_path, filename)
        d = types.ModuleType("config")
        d.__file__ = filename
        with open(filename, mode="rb") as config_file:
            exec(compile(config_file.read(), filename, "exec"), d.__dict__)
        return self.from_object(d)

    def from_object(self, obj):
        for key in dir(obj):
            if key.isupper():
                subdict = super()
                delims = key.split("__")

                for delim in delims[:-1]:
                    if not subdict.__contains__(delim):
                        subdict.__setitem__(delim, {})
                    subdict = subdict.__getitem__(delim)

                subdict.__setitem__(delims[-1], getattr(obj, key))
        return self

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {dict.__repr__(self)}>"
