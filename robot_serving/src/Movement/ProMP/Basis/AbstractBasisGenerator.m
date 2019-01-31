classdef AbstractBasisGenerator
    %ABSTRACTBASISGENERATOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Abstract)
        [basis, basis_d] = generate(obj, Z, dt);
    end
    
end