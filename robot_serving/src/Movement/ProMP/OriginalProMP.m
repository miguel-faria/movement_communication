classdef OriginalProMP < handle & matlab.mixin.Copyable
    %PROMP Probabilistic motor primitive.
    %   A probabilistic motor primitive (ProMP) is a flexible
    %   representation of trajectories. It allows flexible and efficient
    %   modulation of trajectories through operations from probability
    %   theory.
    %
    %   
    
    properties (GetAccess = public, SetAccess = private)
        Y % Matrix with the demonstrations provided
        N % Number of demonstrations provided
        dof % Number of degrees of freedom
        dt
        Sigma_y % Predicted error in the movement

        built % Boolean indicating if the ProMP was already built.
              % This step is required before conditioning.
        phase_generator
        basis_generator
        Z
        Z_d
        basis
        basis_d
        mu_w
        Sigma_w
    end
    
    methods
        function obj = OriginalProMP(Y, dof, dt, Sigma_y)
            % PROMP Returns an instance of ProMP
            %   pmp = OriginalProMP(Y, DOF) produces a ProMP with demonstrated
            %   trajectories Y and number of degrees of freedom DOF.
            %   Y is expected to be ordered by demonstrations and then by
            %   degrees of freedom:
            %   demo1 dof1 | demo1 dof2 | ... | demoN dof1 | demoN dof2 |
            %   Assumes DT = 0.01 and SIGMA_Y = zeros(DOF).
            %   
            %   pmp = OriginalProMP(Y, DOF, DT) is as above, but allows a different
            %   DT. When not provided, DT is assumed to be 0.01.
            %
            %   pmp = OriginalProMP(Y, DOF, DT, SIGMA_Y) is as above but allows a
            %   different SIGMA_Y. When not provided, SIGMA_Y is assumed
            %   to be 0.

            narginchk(2, 4);

            if nargin < 3
                dt = 0.01;
            end
            if nargin < 4
                Sigma_y = zeros(dof);
            end

            assert(isnumeric(Y), 'Y must be a numeric matrix');
            assert(mod(size(Y, 2), dof) == 0, 'size of Y makes no sense.');
            assert(isa(dt, 'double'), 'DT must be a number.');
            assert(dt > 0, 'DT must be positive.');
            assert(isa(dof, 'double') && mod(dof, 1) == 0, ...
                'DOF must be an integer.');
            assert(dof > 0, 'DOF must be positive.');
            assert(all(size(Sigma_y) == [dof, dof]), ...
                'SIGMA_Y must be a square matrix with size DOF.');

            obj.Y = Y;
            obj.N = size(Y, 2) / dof;
            obj.dof = dof;
            obj.dt = dt;
            obj.Sigma_y = Sigma_y;
            obj.built = false;
        end

%         function y = Y(obj, i)
%             % Y Returns the demonstrations provided.
%             %   Y() returns a copy of the demonstrations provided.
%             %
%             %   Y(I) returns a copy of some of the demonstrations provided.
%             %   I is a 1-by-D vector, containing the index of the D
%             %   desired demonstrations.
%             
%             narginchk(1, 2);
%             
%             if nargin < 2
%                 y = obj.Y;
%             elseif nargin < 3
%                 assert(isnumeric(i), 'I should be an array with indexes.');
%                 
%                 idxs = [];
%                 for ii = 1:length(i)
%                     base_idx = (i(ii) - 1) * obj.dof + 1;
%                     idxs = [idxs, base_idx : base_idx + obj.dof - 1];
%                 end
%                 
%                 y = obj.Y(:, idxs);
%             end
%        end
        
        function [w] = build(obj, phase_generator, basis_generator, use_EM)
            % BUILD Builds the ProMP.
            %   Building a ProMP is the necessary first step before using
            %   it. Starting from the demos provided, this method computes
            %   the probabilistic distribution of the trajectory space.
            %
            %   pmp.build(PHASE_GENERATOR, BASIS_GENERATOR) builds the
            %   ProMP using PHASE_GENERATOR as the provider of phase, and
            %   BASIS_GENERATOR as the provider of basis functions.
            %   PHASE_GENERATOR and BASIS_GENERATOR should extend
            %   AbstractPhaseGenerator and AbstractBasisGenerator,
            %   respectively.
            %
            %   pmp.build(PHASE_GENERATOR, BASIS_GENERATOR, USE_EM) the
            %   same as above but allows the parameters computation to be
            %   performed using the EM algorithm. USE_EM should be a
            %   logical value. If USE_EM is true, EM will be used.

            narginchk(3, 4);

            if nargin < 4
                use_EM = false;
            end

            assert(isa(phase_generator, 'AbstractPhaseGenerator'), ...
                'PHASE_GENERATOR must extend AbstractPhaseGenerator.');
            assert(isa(basis_generator, 'AbstractBasisGenerator'), ...
                'BASIS_GENERATOR must extend AbstractBasisGenerator.');

            [obj.Z, obj.Z_d] = phase_generator.generate(obj.dt, 1);
            [obj.basis, obj.basis_d] = basis_generator.generate(obj.Z, obj.dt);

            w = OriginalProMP.computeW(obj.Y, obj.dof, obj.basis);
            [obj.mu_w, obj.Sigma_w] = OriginalProMP.computeDistributionParams(w);

            obj.built = true;
        end

        function condition(obj, y, t, Sigma_new)
            % CONDITION Conditions the ProMP to a specific point.
            %   pmp.condition(Y) conditions the ProMP to finish at point Y.
            %   Y should be a column array, respecting the DOF of the
            %   demonstrations provided.
            %
            %   pmp.condition(Y, T) conditions the ProMP to point Y at time
            %   T. T should should be interpreted as a percentage of the
            %   trajectory - T = 0.5 implies that the trajectory should be
            %   conditioned to Y at its middlepoint.
            %
            %   pmp.condition(Y, T, SIGMA_NEW) is as above, but allows the
            %   specification of the uncertainty regarding the point we are
            %   conditioning upon. SIGMA_NEW should be a symmetric matrix,
            %   with dimensions respecting the number of DOF of the
            %   demonstrations provided. If not provided, SIGMA_NEW is
            %   assumed to be 0.
            
            narginchk(2, 4);
            
            if nargin < 3
                t = 1;
            end
            if nargin < 4
                Sigma_new = zeros(obj.dof);
            end
            
            assert(obj.built == true, ...
                'ProMP must be built before conditioning.');
            assert(isnumeric(y), 'Y must be a numeric array.');
            assert(size(y, 1) == obj.dof, 'Y must be column array.');
            assert((t >= 0) &&  (t <= 1), ...
                'T must be in the interval [0, 1]');
            assert(isnumeric(Sigma_new), ...
                'SIGMA_NEW must be numeric matrix');
            assert(all(size(Sigma_new) == [obj.dof, obj.dof]), ...
                'SIGMA_NEW must be a square matrix with size DOF.');
        
            t_index = max(round(t / obj.dt), 1);

            basis_t = obj.basis(:, t_index);
            basis_t = kron(eye(obj.dof), basis_t);

            tmp = obj.Sigma_w * basis_t * pinv(Sigma_new + basis_t' * obj.Sigma_w * basis_t);
            obj.mu_w = obj.mu_w + tmp * (y - basis_t' * obj.mu_w);
            obj.Sigma_w = obj.Sigma_w - tmp * basis_t' * obj.Sigma_w;
        end
        
        function [new] = conditionNonDestructive(obj, varargin)
            % CONDITIONNONDESTRUCTIVE Conditions and returns a copy of
            % the ProMP.
            %
            %   Works the same way as CONDITION, but does not modify the
            %   ProMP. Returns a copy instead.
            %
            %   See also ProMP.CONDITION
            
            new = obj.copy();
            new.condition(varargin{:});
        end
        
        function Y = mostProbable(obj)
            % MOSTPROBABLE Returns the most probable trajectory according
            % to this ProMP.

            narginchk(1, 1);

            assert(obj.built == true, ...
                'ProMP must be built before computing the most probable trajectory.');

            time = obj.Z;
            
            basis = kron(eye(obj.dof), obj.basis);
            
            Y = basis' * obj.mu_w;
            Y = reshape(Y, [], obj.dof);
            Y = [time Y];
        end
            
        function Y = random(obj, n)

            narginchk(1, 2);

            if nargin < 2
                n = 1;
            end

            assert(obj.built == true, ...
                'ProMP must be built before computing random trajectory(ies).');
            assert(isa(n, 'double') && mod(n, 1) == 0, ...
                'N must be an integer.');
            assert(n > 0, 'N must be positive.');

            w = mvnrnd(obj.mu_w', obj.Sigma_w, n)';

            time = obj.Z;

            Y = kron(eye(obj.dof), obj.basis)' * w;
            Y = reshape(Y, [], obj.dof * n);
            Y = [time, Y]; % add time column
            
            
%             Y = zeros(length(time), n * obj.dof + 1);
%             Y(:, 1) = time;
%            
%             for it = 1:length(time)
%                 basis_t = obj.basis(:, it);
%                 dof_basis_t = kron(eye(obj.dof), basis_t);
%                 
%                 Y(it, 2:end) = (dof_basis_t' * ws_new)';
%             end
        end
        
        function prob = probability(obj, Y, normalize)
            
            narginchk(2, 3);
            
            if nargin < 3
                normalize = false;
            end

            assert(obj.built == true, ...
                'ProMP must be built before computing the probability of a trajectory.');            
            assert(mod(size(Y, 2), obj.dof) == 0, ...
                'Y does not respect the DOF of the demonstrations provided.');
            assert(isnumeric(Y), 'Y must be a numeric matrix.');
            assert(islogical(normalize), ...
                'NORMALIZE must be a logical value.');
            
            w = OriginalProMP.computeW(Y, obj.dof, obj.basis);
            prob = mvnpdf(w', obj.mu_w', obj.Sigma_w);
            %prob = ugaussian(w', obj.mu_w, obj.Sigma_w);
            
            if normalize
                den = mvnpdf(obj.mu_w', obj.mu_w', obj.Sigma_w);
                %den = ugaussian(obj.mu_w', obj.mu_w, obj.Sigma_w);
                prob = prob ./ den;
            end
        end
    end

    methods (Static = true)
        function [w] = computeW(Y, dof, basis)

            narginchk(3, 3);

            N = size(Y, 2) / dof;

            dof_basis = kron(eye(dof), basis);
            Y_merged_dofs = reshape(Y, [], N);

            w = dof_basis' \ Y_merged_dofs;
        end

        function [mu_w, Sigma_w] = computeDistributionParams(w)

            narginchk(1, 1);

            mu_w = mean(w, 2);
            Sigma_w = cov(w') + eye(length(mu_w)) * 0.00001;
        end
    end
end

