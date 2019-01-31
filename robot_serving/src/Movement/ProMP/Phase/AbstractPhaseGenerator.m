classdef (Abstract) AbstractPhaseGenerator
    properties
    end
    
    methods (Abstract)
        [phase, phase_d] = generate(obj, dt, T);
    end
    
end

