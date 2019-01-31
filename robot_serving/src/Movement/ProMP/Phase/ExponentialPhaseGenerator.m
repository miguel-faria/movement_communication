classdef ExponentialPhaseGenerator < AbstractPhaseGenerator
    properties (Access = private)
        alpha_
    end
    
    methods
        function obj = ExponentialPhaseGenerator(alpha)
            if nargin < 1
                alpha = 4;
            end
            
            obj.alpha_ = alpha;
        end
        
        function [phase, phase_d] = generate(obj, dt, T)
            if nargin < 2
                error('ExponentialPhaseGenerator: generate: requires dt.');
            end
            if nargin < 3
                T = 1;
            end

            phase = exp(-obj.alpha_ * linspace(0, T, T / dt))';
            phase_d = diff(phase);
            phase_d = [phase_d; phase_d(end)];
        end
    end
    
end

