classdef NormalizedGaussianBasisGenerator < AbstractBasisGenerator
    properties (GetAccess = public, SetAccess = private)
        N % Number of basis functions
    end
    
    methods
        function obj = NormalizedGaussianBasisGenerator(N)
            obj.N = N;
        end
        
        function [basis, basis_d] = generate(obj, Z, dt)
            mu = linspace(0, 1, obj.N)';
            sigma = ones(obj.N, 1) / obj.N;

            Z_mu = bsxfun(@minus, Z, mu');
            Z_mu_sigma = bsxfun(@times, Z_mu, 1 ./ sigma');
            
            basis = bsxfun(@times, exp(-0.5 * Z_mu_sigma .^ 2), 1 ./ sigma' / sqrt(2 * pi));
            basis_sum = sum(basis, 2);
            
            basis = bsxfun(@times, basis, 1 ./ basis_sum); % normalize
            basis = basis';
            
            basis_d = diff(basis, 1, 2) / dt;
            basis_d = [basis_d, basis_d(:, end)]; % make same size as basis
        end
    end
    
end