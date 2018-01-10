%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                       %  
%%% Instituto Federal do Ceara, Campus Maracanau                          %
%%% Bacharelado em Ciencia da Computacao                                  %
%%% Disciplina: Redes Neurais Artificiais, Prof. Ajalmar Rocha            %
%%% Aluno: Jose Igor de Carvalho                                          % 
%%%                                                                       % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;
format short

REALIZATIONS = 20;

% Carregando o conjunto de dados
dataset = load('C:\Users\This\Desktop\rna-e-ia\Datasets\iris.txt');

% Normalizando o conjunto de dados
dataset = normalize_melhorada(dataset);

% Permutando linhas do conjunto de dados
dataset = dataset(randperm(size(dataset,1)),:);

% Os dados s�o capturados aleatoriamente visto que as linhas s�o
% permutadas
% Conjunto de treinamento
training_set = dataset(1:size(dataset, 1)-30, :);
% Conjunto de testes
test_set = dataset (size(dataset, 1)-29:end, :);

%Capturando apenas os atributos normalizados
x = dataset(:, 1:size(dataset, 2)-1);
%Capturando apenas as tags associadas a cada atributo
y = dataset (:,size(dataset,2));

% Vetor criado para armazenar a taxa de acerto do perceptron com o cojunto
% de testes em cada realiza��o
match_rate = zeros(REALIZATIONS, 1);
results = zeros(size(test_set, 2), 1);

% Calculando a matriz de confus�o para o ultima realiza��o
k = horzcat(-ones(size(test_set, 1), 1), test_set(:,1:4));

for i=1:REALIZATIONS
    
    % Treinando o neuronio com o conjutno de treino
    w = learning_rule(training_set, 20, 0.1);
    % Capturando a quantidade de acertos e erros para o neuronio treinado
    [matches, not_matches] = verificar(test_set, w);
    verificar(test_set, w);
    % Calculando a taxa de acerto para cada realiza��o
    match_rate(i) = matches/(matches+not_matches);
    for j=1:size(test_set, 1)
        % Calculando o produto interno entre as entradas do neuronio e o vertor
        % de pesos
        u = dot (k(j,:), w);
        % Armazenando as sa�das do conjunto de treinamento
        results(j) = activation(u);
    end
    % Mostrando a matriz de confus�o para cada realiza��o
    fprintf ('REALIZATION %d', i);
   confusion_matrix =  confusionmat(results, test_set(:,5))
end

% % Calculando a matriz de confus�o para o ultima realiza��o
% k = horzcat(-ones(size(test_set, 1), 1), test_set(:,1:4));

for i=1:size(test_set, 1)
    % Calculando o produto interno entre as entradas do neuronio e o vertor
    % de pesos
    u = dot (k(i,:), w);
    % Armazenando as sa�das do conjunto de treinamento
    results(i) = activation(u);
end

% Calculando a matriz de confus�o
accuracy = sum(match_rate)/size(match_rate,1); % Tamb�m � possivel dividir por REALIZATIONS
standard_deviation = std(match_rate);
l_confusion_matrix = confusionmat(results, test_set(:,5));

disp ('--------- DADOS ----------')
accuracy
standard_deviation
l_confusion_matrix
disp ('--------------------------')

%% PLOTANDO GRAFICO DA TAXA DE ACERTOS
figure(1)

plot(match_rate, 'b', 'LineStyle', '--')
hold on
grid on
scatter(1:REALIZATIONS, match_rate, 'black')

xlim([1 20])
title('Match rate')
hold off


%% PLOTANDO O MAPA DE CORES
% % FONTE: https://pastebin.com/TfXR1Lmt

% xrange = [0 1.0];
% yrange = [0 1.0];
% inc = 0.09;
% 
% [x_coord, y_coord] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
% 
% xy = [x_coord(:) y_coord(:)];
% sizeXY = size(xy);
% lines = sizeXY(1);
% 
% figure(4)
% for i=1:lines
%     if((w(1)*-1+w(2)*xy(i,1)+w(3)*xy(i,2)) > 0); %-1 � do baias
%         scatter(xy(i,1),xy(i,2),'.g');
%        hold on; grid on;
%     elseif((w(1)*-1+w(2)*xy(i,1)+w(3)*xy(i,2)) < 0);
%        scatter(xy(i,1),xy(i,2),'.r');
%        hold on; grid on;
%     end
% end
% %legend('Class 1', 'Class 2');
% title('Colormap')
% xlim([0 1])
% ylim([0 1])
% hold off
%% FIM
