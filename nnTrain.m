function layers = nnTrain(layers, opt, trainData, trainProbability)
    trainNum = size(trainData, 2);
    batchNum = trainNum/opt.batchSize;
    if rem(batchNum, 1)
        error('Batch number is not an integer!');
    end
    k = randperm(trainNum);
    for a = 1:batchNum
        layers = nnFF(layers, trainData(:, k((a-1)*opt.batchSize+1 : a*opt.batchSize)));
        layers = nnBP(layers, opt, trainData(:, k((a-1)*opt.batchSize+1 : a*opt.batchSize)), trainProbability(:, k((a-1)*opt.batchSize+1 : a*opt.batchSize)));
    end
end