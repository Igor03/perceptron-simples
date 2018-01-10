function [normalized_dataset] = normalize(dataset)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    
    inputs = dataset(:, 1:size(dataset, 2)-1);
    tags = dataset(:,size(dataset, 2));
    
%     max_values = [];
%     min_values = [];
    
    for i=1:size(inputs, 2)
        max_values(i) = max(inputs(:, i:size(inputs, 2)-(size(inputs, 2)-i)));
        min_values(i) = min(inputs(:, i:size(inputs, 2)-(size(inputs, 2)-i)));
    end
    
    inputs(:, 1:size(inputs, 2)-1);
    inputs(:, 2:size(inputs, 2));
    
    %max_values
    %min_values
    
    for i=1:size(inputs, 2)
        for j=1:size(inputs, 1)
            inputs(j, i) = (inputs(j, i)-min_values(i))/(max_values(i)-min_values(i));
        end
    end
    
    normalized_dataset = horzcat(inputs, tags);

end