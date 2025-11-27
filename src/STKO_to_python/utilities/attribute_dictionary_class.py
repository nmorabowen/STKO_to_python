class AttrDict(dict):
    """
    A dictionary that allows attribute-style access.
    
    Example:
    >>> ad = AttrDict()
    >>> ad['key'] = 'value'
    >>> ad.key
    'value'
    >>> ad.new_key = 'new_value'
    >>> ad['new_key']
    'new_value'
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, name):
        """Allow ad.key to access ad['key']."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Allow ad.key = 'value' to set ad['key'] = 'value'."""
        self[name] = value