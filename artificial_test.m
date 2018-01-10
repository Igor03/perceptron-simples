%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                       %
%%% Instituto Federal do Ceara, Campus Maracanau                          %
%%% Bacharelado em Ciencia da Computacao                                  %
%%% Disciplina: Redes Neurais Artificiais, Prof. Ajalmar Rocha            %
%%% Aluno: Jose Igor de Carvalho                                          %
%%%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Idéia para a criação do conjunto de dados linearmente separado: Fonte:
% http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/nn03_perceptron/

clc; clear; close all;
REALIZATIONS = 5;
%% GERANDO DADOS ALEATORIAMENTE

N = 500; %Quantidade de pontos à serem gerados

offset = 5; % Distancia media entre cada ponto a ser gerado

x = [randn(2,N) randn(2,N)+offset]; % atributos
y = [zeros(1,N) ones(1,N)];         % tags

% Criando e normalizando um conjunto da dados artificial linearmento
% separável
dataset = horzcat(x', y');

% Normalizando o conjunto de dados
%dataset = normalize(dataset);

dataset = normalize_melhorada(dataset);

% Permuntando as linhas do conjunto de dados
dataset = dataset(randperm(size(dataset,1)),:);

%Capturando apenas os atributos normalizados
x = dataset(:, 1:size(dataset, 2)-1);
%Capturando apenas as tags associadas a cada atributo
y = dataset (:,size(dataset,2));

%% PLOTANDO OS DADOS INICIAIS
figure(1)
%plotpv(x', y');
gscatter(x(:,1), x(:,2), y,'rgb','osd')

% Configurações do plot
title('Data')
legend('Class 1', 'Class 2')
%legend('Class 1', 'Class 2','Location','NorthOutside', ...
 %   'Orientation', 'horizontal');
xlabel('x');
ylabel('y');
xlim([0 1])
ylim([0 1])


% Conjunto de treinamento
training_set = dataset(1:size(dataset, 1)-300, :);

% Conjunto de testes
test_set = dataset (size(dataset, 1)-299:end, :);


match_rate = zeros(REALIZATIONS, 1);
results = zeros(size(test_set, 2), 1);
k = horzcat(-ones(size(test_set, 1), 1), test_set(:,1:2));

for i=1:REALIZATIONS
    
    w = learning_rule(training_set, 10, 0.1);
    [matches, not_matches] = verificar(test_set, w);
    verificar(test_set, w);
    match_rate(i) = matches/(matches+not_matches);
    for j=1:size(test_set, 1)
        % Calculando o produto interno entre as entradas do neuronio e o vertor
        % de pesos
        u = dot (k(j,:), w);
        % Armazenando as saídas do conjunto de treinamento
        results(j) = activation(u);
    end
    % Mostrando a matriz de confusão para cada realização
    fprintf ('REALIZATION %d', i);
    confusion_matrix = confusionmat(results, test_set(:,3))
    
end

%% PLOTANDO OS DADOS INICIAIS JUNTO COM A RETA QUE OS CLASSIFICA
figure(2)
gscatter(x(:,1), x(:,2), y,'rgb','osd')

hold on

bias = w(1);
w1 = w(2);
w2 = w(3);

t = -1: .5: 5;
y1 = -(w1/w2)*t + (bias/w2);

plot(t, y1, 'black');

% Configurações do plot
title('Classification')
legend('Class 1', 'Class 2')
%legend('Class 1', 'Class 2','Location','NorthOutside', ...
%    'Orientation', 'horizontal');
xlabel('x');
ylabel('y');
xlim([0 1])
ylim([0 1])

hold off

%% PLOTANDO O MAPA DE CORES
% FONTE: https://pastebin.com/TfXR1Lmt

xrange = [0 1.0];
yrange = [0 1.0];
inc = 0.05;

[x_coord, y_coord] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

xy = [x_coord(:) y_coord(:)];
sizeXY = size(xy);
lines = sizeXY(1);

figure(4)
for i=1:lines
    if((w(1)*-1+w(2)*xy(i,1)+w(3)*xy(i,2)) > 0); %-1 é do baias
        scatter(xy(i,1),xy(i,2),'.g');
       hold on; grid on;
    elseif((w(1)*-1+w(2)*xy(i,1)+w(3)*xy(i,2)) < 0);
       scatter(xy(i,1),xy(i,2),'.r');
       hold on; grid on;
    end
end
legend('Class 1', 'Class 2');
title('Colormap')
xlim([0 1])
ylim([0 1])
hold off

%% Calculando acurácia
accuracy = sum(match_rate)/size(match_rate,1); % Também é possivel dividir por REALIZATIONS
%% Calculando desvio padrão
standard_deviation = std(match_rate);
%% Calculando a matriz de confusão para ultima realização
l_confusion_matrix = confusionmat(results, test_set(:,3));

%% Mostrando os dados na tela
disp ('--------- DADOS ----------')
accuracy
standard_deviation
l_confusion_matrix
disp ('--------------------------')

%% PLOTANDO GRAFICO DA TAXA DE ACERTOS
figure(3)

plot(match_rate, 'b', 'LineStyle', '--')
hold on
grid on
scatter(1:REALIZATIONS, match_rate, 'black')

xlim([1 20])
title('Match rate')
hold off