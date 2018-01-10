function [matches, not_matches] = verificar(dataset, pesos)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    m = 0; %Contador de acertos
    nm = 0; % Contador de erros
    
    % Concatenando com -1's
    dataset = horzcat(-ones(size(dataset, 1), 1), dataset);

    k = dataset(:, 1:size(dataset, 2)-1); % Atributos
    p = dataset(:,size(dataset, 2)); % Tags
    
    
    for i=1:size(dataset, 1)
        % calculandoo erro (Saida_obtida - Saida_desejada)
        if (activation (dot(k(i,:), pesos)) - p(i) ~= 0)
            nm = nm + 1;
        else
            m = m + 1;
        end
    end
    matches = m;
    not_matches = nm;
end

