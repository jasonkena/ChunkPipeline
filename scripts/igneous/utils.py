from joblib.externals.loky import get_reusable_executor


class DotDict(dict):
    # modified from https://stackoverflow.com/a/13520518/10702372
    """
    A dictionary that supports dot notation as well as dictionary access notation.
    Usage: d = DotDict() or d = DotDict({'val1':'first'})
    Set attributes: d.val2 = 'second' or d['val2'] = 'second'
    Get attributes: d.val2 or d['val2']

    NOTE: asserts that dictionary does not contain tuples (YAML)
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = self._convert_list(value)
            self[key] = value

    def _convert_list(self, lst):
        new_list = []
        for item in lst:
            if isinstance(item, dict):
                new_list.append(DotDict(item))
            elif isinstance(item, list):
                new_list.append(self._convert_list(item))
            else:
                new_list.append(item)
        return new_list

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
