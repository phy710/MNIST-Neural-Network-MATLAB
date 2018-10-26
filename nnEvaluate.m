function [l, acc] = nnEvaluate(layers, opt, x, d)
    layers = nnFF(layers, x);
    y = layers{end}.output;
    l = loss(d, y, opt);
    [~, estimatedLabels] = max(y);
    [~, labels] = max(d);
    acc = mean(estimatedLabels==labels);
end