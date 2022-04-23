# Module whos sole purpose is to define the `@Cach` decorator. This enables methods to be use like class atributes, i.e. with the cls.attribute syntax, evalutes the attribute the first time it is invoked, and then caches the result. Thus getting rid of reevaluating expensive calculations again.

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


# TODO: rename Cach into Cache

  # def Eigvecs_Hu(self):
    #     # H = [scipy.sparse.csr_matrix(self.H(u, 1)) for u in self.u_array]
    #     H = np.array([self.H(u, 1) for u in self.u_array])
    #     eigvals, eigvecs = np.linalg.eigh(H)
    #     # idx = np.argsort(eigvals, axis=1)
    #     # print(eigvals.shape, eigvecs.shape)
    #     # print(np.round(eigvals), "\n", np.round(eigvecs))
    #     # eigvals = np.take_along_axis(eigvals, idx, axis=1)
    #     # eigvecs = np.take_along_axis(eigvecs, idx[..., None], axis=1)
    #     # eigvecs = eigvecs[:, ind, :]  # second axis !!
    #     return eigvecs[..., 0]
    #     # return H
