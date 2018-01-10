function [result] = activation (u)
    if (u<0)
        result = 0;
        return;
    end
    result = 1;
    return;
end
