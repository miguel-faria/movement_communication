import numpy as np
import scipy as sp
import scipy.stats
import copy


class ProMP(object):
    def __init__(self, Y, dof, dt=0.01, Sigma_y=None):
        if Sigma_y is None:
            Sigma_y = np.zeros((dof, dof))
        
        if not isinstance(Y, np.ndarray):
            raise ValueError('Y must be numpy array')
        if Y.shape[1] % dof != 0:
            raise ValueError('Y shape incosistent with number of dofs')
        if dt <= 0:
            raise ValueError('dt must be positive number')
        if Sigma_y.shape != (dof, dof):
            raise ValueError('Sigma_y must be a square matrix with size dof')
        if not np.all(Sigma_y == Sigma_y.T):
            raise ValueError('Sigma_y must be symmetric matrix')
        
        self._Y = Y
        self._N = Y.shape[1] / dof
        self._dof = dof
        self._dt = dt
        self._Sigma_y = Sigma_y
        self._built = False
    
    def build(self, phase_generator, basis_generator):
        if not callable(phase_generator):
            raise ValueError('phase_generator must be callable')
        if not callable(basis_generator):
            raise ValueError('basis_generator must be callable')
        
        self._Z = phase_generator(self._dt, 1)
        self._basis = basis_generator(self._Z, self._dt)
        
        w = ProMP.compute_w(self._Y, self._dof, self._basis)
        self._mu_w, self._Sigma_w = ProMP.compute_distribution_params(w)
        
        self._built = True
        
    def condition(self, y, t=1, Sigma_new=None):
        if Sigma_new is None:
            Sigma_new = np.zeros((self._dof, self._dof))
            
        if not self._built:
            raise ValueError('ProMP must be built before conditioning')
        if not isinstance(y, np.ndarray):
            raise ValueError('y must be numpy array')
        if y.shape[0] != self._dof:
            raise ValueError('y must be column array')
        if t < 0 or t > 1:
            raise ValueError('t must be in the interval [0, 1]')
        if Sigma_new.shape != (self._dof, self._dof):
            raise ValueError('Sigma_new must be a square matrix with size dof')
        if not np.all(Sigma_new == Sigma_new.T):
            raise ValueError('Sigma_new must be symmetric matrix')
            
        t_index = max(round(1.0 * t / self._dt) - 1, 0)
        
        basis_t = self._basis[:, t_index][:, np.newaxis]
        basis_t = np.kron(np.eye(self._dof), basis_t)
        
        tmp = self._Sigma_w.dot(basis_t.dot(np.linalg.pinv(Sigma_new + basis_t.T.dot(self._Sigma_w.dot(basis_t)))))
        self._mu_w = self._mu_w + tmp.dot(y - basis_t.T.dot(self._mu_w))
        self._Sigma_w = self._Sigma_w - tmp.dot(basis_t.T.dot(self._Sigma_w))
        
    def condition_non_destructive(self, *args):
        new = copy.deepcopy(self)
        new.condition(*args)
        
        return new
        
    def most_probable(self):
        if self._built != True:
            raise ValueError('ProMP must be built before computing ', 
                             'most probable trajectory')
        
        time = self._Z
        
        basis = np.kron(np.eye(self._dof), self._basis)
        
        Y = basis.T.dot(self._mu_w)
        Y = np.reshape(Y, (-1, self._dof), order='F')
        
        Y = np.hstack((time, Y))
        
        return Y
    
    def random(self, n=1):
        if n < 0:
            raise ValueError('n must be positive')
        if self._built != True:
            raise ValueError('ProMP must be built before computing '
                             'random trajectory')
            
        w = np.random.multivariate_normal(self._mu_w.flatten(), self._Sigma_w, n).T
        
        time = self._Z
        
        basis = np.kron(np.eye(self._dof), self._basis)
        Y = basis.T.dot(w)
        Y = np.reshape(Y, (-1, self._dof * n), order='F')
        Y = np.hstack((time, Y))
        
        return Y
    
    def probability(self, Y, normalize=False):
        if self._built != True:
            raise ValueError('ProMP must be built before computing probability')
        if Y.shape[1] % self._dof != 0:
            raise ValueError('Y does not respect the dof of the demos provided')
    
        w = ProMP.compute_w(Y, self._dof, self._basis)
        prob = sp.stats.multivariate_normal.pdf(w.T,
                                                self._mu_w.flatten(),
                                                self._Sigma_w)
        print 'prob: ', prob
        
        if normalize:
            den = sp.stats.multivariate_normal.pdf(self._mu_w.flatten(), 
                                                   self._mu_w.flatten(), 
                                                   self._Sigma_w)
            prob = prob / den
    
        return prob
    
    @staticmethod
    def compute_w(Y, dof, basis):
        N = Y.shape[1] / dof
        dof_basis = np.kron(np.eye(dof), basis)
        Y_merged_dofs = np.reshape(Y, (-1, N), order='F')
        
        w, _, _, _ = np.linalg.lstsq(dof_basis.T, Y_merged_dofs)
        
        return w
        
    @staticmethod
    def compute_distribution_params(w):
        mu_w = np.mean(w, axis=1, keepdims=True)
        
        Sigma_w = np.cov(w) + np.eye(mu_w.shape[0]) * 0.00001
        
        return mu_w, Sigma_w
