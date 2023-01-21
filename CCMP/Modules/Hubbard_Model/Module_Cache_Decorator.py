# Module whos sole purpose is to define the `@Cach` decorator. This enables methods to be used like class atributes, i.e. with the cls.attribute syntax, evalutes the attribute the first time it is invoked, and then caches the result. Thus getting rid of reevaluating expensive calculations again.

import functools


class CachedAttribute(object):
    ''' Computes attribute value and caches it in the instance. '''

    def __init__(self, method, name=None):
        # record the unbound-method and the name
        self.method = method
        self.name = name or method.__name__

    def __get__(self, inst, cls):
        if inst is None:
            # instance attribute accessed on class, return self
            return self
        # compute, cache and return the instance's attribute value
        result = self.method(inst)
        setattr(inst, self.name, result)
        return result


def Cach(func):
    return functools.wraps(func)(CachedAttribute(func))
