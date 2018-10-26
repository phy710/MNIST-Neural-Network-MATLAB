function [layers, opt] = nnSetup()
    opt.eta = 0.5;
    opt.loss = 'MSE';
    opt.batchSize = 100;
    opt.L2 = 0.001;
    layers{1}.w = rand(100, 784, 'gpuArray')-0.5;
    layers{1}.act = 'tanh';
    layers{2}.w = rand(100, size(layers{1}.w, 1), 'gpuArray')-0.5;
    layers{2}.act = 'tanh';
    layers{3}.w = rand(10, size(layers{2}.w, 1), 'gpuArray')-0.5;
    layers{3}.act = 'sigmoid';
    for a = 1 : numel(layers)
        layers{a}.b = rand(size(layers{a}.w, 1), 1, 'gpuArray')-0.5;
    end
end