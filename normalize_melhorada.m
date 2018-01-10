%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                       %
%%% Instituto Federal do Ceara, Campus Maracanau                          %
%%% Bacharelado em Ciencia da Computacao                                  %
%%% Disciplina: Redes Neurais Artificiais, Prof. Ajalmar Rocha            %
%%% Aluno: Jose Igor de Carvalho                                          %
%%%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [normalized_dataset] = normalize_melhorada(dataset)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    n = size(dataset, 1);
    min_n = repmat(min(dataset), n, 1);
    max_n = repmat(max(dataset), n, 1);
    normalized_dataset = (dataset - min_n) ./ (max_n-min_n);

end

