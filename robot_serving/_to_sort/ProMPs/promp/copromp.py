import numpy as np
import scipy as sp
from promp import ProMP


class CoProMP(ProMP):
    def __init__(self, O, Y, o_dof, dof, o_dt=1, dt=0.01, Sigma_y=None):
        super(CoProMP, self).__init__(Y, dof, dt, Sigma_y)

        if not isinstance(O, np.ndarray):
            raise ValueError('O must be numpy array')
        if O.shape[1] % o_dof != 0:
            raise ValueError('O shape incosistent with number of o_dofs')
        if o_dt <= 0:
            raise ValueError('o_dt must be positive number')
        
        self._O = O
        self._o_dof = o_dof
        self._o_dt = o_dt
    
    def build(self, o_phase_generator, o_basis_generator,
              y_phase_generator, y_basis_generator):
        if not callable(o_phase_generator):
            raise ValueError('o_phase_generator must be callable')
        if not callable(o_basis_generator):
            raise ValueError('o_basis_generator must be callable')
        if not callable(y_phase_generator):
            raise ValueError('y_phase_generator must be callable')
        if not callable(y_basis_generator):
            raise ValueError('y_basis_generator must be callable')
        
        self._o_Z = o_phase_generator(self._o_dt, 1)
        self._o_basis = o_basis_generator(self._o_Z, self._o_dt)
        self._Z = y_phase_generator(self._dt, 1)
        self._basis = y_basis_generator(self._Z, self._dt)
        
        o_w = CoProMP.compute_w(self._O, self._o_dof, self._o_basis)
        y_w = CoProMP.compute_w(self._Y, self._dof, self._basis)
        
        w = np.vstack((o_w, y_w))
        
        self._o_mu_w, self._o_Sigma_w = CoProMP.compute_distribution_params(w)
        self._mu_w, self._Sigma_w = CoProMP.compute_distribution_params(y_w)
        
        self._built = True
        
    def condition(self, o, t=1, o_Sigma_new=None, y_Sigma_new=None):
        # TODO:
        # - support two ways inference: o -> c, c -> o
        #   - for now, only o -> c
        if o_Sigma_new is None:
            o_Sigma_new = np.zeros((self._o_dof, self._o_dof))
        if y_Sigma_new is None:
            y_Sigma_new = np.zeros((self._dof, self._dof))
            
        if not self._built:
            raise ValueError('ProMP must be built before conditioning')
        if not isinstance(o, np.ndarray):
            raise ValueError('o must be numpy array')
        if o.shape[0] != self._o_dof:
            raise ValueError('y must be column array')
        if o_Sigma_new.shape != (self._o_dof, self._o_dof):
            raise ValueError('o_Sigma_new must be a square matrix with size o_dof')
        if not np.all(o_Sigma_new == o_Sigma_new.T):
            raise ValueError('o_Sigma_new must be symmetric matrix')
        if y_Sigma_new.shape != (self._dof, self._dof):
            raise ValueError('y_Sigma_new must be a square matrix with size dof')
        if not np.all(y_Sigma_new == y_Sigma_new.T):
            raise ValueError('y_Sigma_new must be symmetric matrix')
        
        # Compute t idxs
        o_t_index = np.max(round(1.0 * t / self._o_dt) - 1, 0)        
        t_index = np.max(round(1.0 * t / self._dt) - 1, 0)
        
        # Compute Sigma_new
        Sigma_new = sp.linalg.block_diag(o_Sigma_new, y_Sigma_new)
        
        # Compute basis
        o_basis_t = self._o_basis[:, o_t_index][:, np.newaxis]
        o_basis_t = np.kron(np.eye(self._o_dof), o_basis_t)
        
        basis_t = self._basis[:, t_index][:, np.newaxis]
        basis_t = np.kron(np.eye(self._dof), basis_t)  # hackish way of computing basis size...
        basis_t = np.zeros(basis_t.shape)
        
        # Compute H (as in [1])
        H = sp.linalg.block_diag(o_basis_t, basis_t)
        
        # Extend the new o
        o = np.vstack((o, np.zeros((self._dof, 1))))
        
        # Update mu_w and Sigma_w
        tmp = self._o_Sigma_w.dot(H.dot(np.linalg.pinv(Sigma_new + H.T.dot(self._o_Sigma_w.dot(H)))))
        self._o_mu_w = self._o_mu_w + tmp.dot(o - H.T.dot(self._o_mu_w))
        self._o_Sigma_w = self._o_Sigma_w - tmp.dot(H.T.dot(self._o_Sigma_w))
        
        self._mu_w = self._o_mu_w[o_basis_t.shape[0]:, 0][:, np.newaxis]
        self._Sigma_w = self._o_Sigma_w[o_basis_t.shape[0]:, o_basis_t.shape[0]:]

    def condition_non_destructive(self, *args):
        new = copy.deepcopy(self)
        new.condition(*args)
        
        return new
