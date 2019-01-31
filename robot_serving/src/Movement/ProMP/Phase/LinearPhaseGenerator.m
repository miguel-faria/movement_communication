classdef LinearPhaseGenerator < AbstractPhaseGenerator
    properties
    end
    
    methods
        function [phase, phase_d] = generate(obj, dt, T)
        % GENERATE Generates a linear phase.
        %   phase = generate(DT) returns a column array with the linear
        %   phase, starting at 0 and ending with 1, with steps DT.
        %
        %   phase = generate(DT, T) returns the linear phase, starting
        %   at 0 and ending in T, with steps DT.
        
            if nargin < 2
                error('LinearPhaseGenerator: generate: requires dt.');
            end
            if nargin < 3
                T = 1;
            end

            phase = linspace(0, T, T / dt)';
            phase_d = diff(phase);
            phase_d = [phase_d; phase_d(end)];
        end
    end
    
end

