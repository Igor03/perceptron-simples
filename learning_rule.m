function [synaptic_weights] = learning_rule(dataset, epochs, learning_rate)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    % Retorna um vetor de pesos sinápticos
    %dataset = load(dataset);
    %dataset = dataset;
    training_set = horzcat(-ones(size(dataset, 1), 1), dataset);
    
    % Aqui definimos uma margem para geracao de um vetor de pesos
    % aleatorios
    w_min = -0.5; % Peso aleatorio minimo igual a -0.5
    w_max = 0.5; % Peso aleatorio maximo igual a 0.5
    
    % Definimos o tamanho do vetor de pesos sinapitocos
    n = size(training_set,2)-1; 
    
    % Gerando vetor de pesos aleatoriamente
    synaptic_weights = w_min+rand(n, 1)*(w_max - w_min); 
    %synaptic_weights = [1.5; 1; 1];
    
    for j=1:epochs
        % Permutando as linhas do conjunto de dados
        dataset = dataset(randperm(size(dataset,1)),:);
        training_set =  horzcat(-ones(size(dataset, 1), 1), dataset);
        
        %Capturando apenas os atributos
        k = training_set(:, 1:size(training_set, 2)-1);
        
        %Capturando apenas as tags associadas a cada  atributo
        p = training_set (:,size(training_set,2));
        
        for i=1:size(training_set, 1)
            u = dot (k(i,:), synaptic_weights);
            synaptic_weights = synaptic_weights + (learning_rate * (p(i)-activation(u))*k(i,:))';
        end
    end
end
