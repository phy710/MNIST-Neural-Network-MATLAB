clear;
clc;
close all;
[layers, opt] = nnSetup();
epochMax = 50;
etaDecay = 0.9;
trainData = loadMNISTImages('train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');
testData = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
trainNum = numel(trainLabels);
testNum = numel(testLabels);
trainProbabilities = zeros(10, trainNum);
for a = 1 : trainNum
    trainProbabilities(trainLabels(a)+1, a) = 1;
end
testProbabilities = zeros(10, testNum);
for a = 1 : testNum
    testProbabilities(testLabels(a)+1, a) = 1;
end
tic;
% evaluate
[trainLoss, trainAcc] = nnEvaluate(layers, opt, trainData, trainProbabilities);
[testLoss, testAcc] = nnEvaluate(layers, opt, testData, testProbabilities);
epoch = 0;
while trainLoss(end)>0 && epoch<epochMax
    epoch = epoch+1;
    layers = nnTrain(layers, opt, trainData, trainProbabilities);
    % evaluate
    [trainLoss(epoch+1), trainAcc(epoch+1)] = nnEvaluate(layers, opt, trainData, trainProbabilities);
    [testLoss(epoch+1), testAcc(epoch+1)] = nnEvaluate(layers, opt, testData, testProbabilities);
    disp(['Epoch: ' num2str(epoch) ', Learning rate: ' num2str(opt.eta) ', Loss: ' num2str(trainLoss(end)) '/' num2str(testLoss(end)) ', Accuracy: ' num2str(trainAcc(end)) '/' num2str(testAcc(end))]);
    if trainLoss(epoch+1)>trainLoss(epoch)
        % If loss increases, decrease learning rate
        opt.eta = opt.eta*etaDecay;
    end
end
toc;
disp(['Training Accuracy: ' num2str(100*trainAcc(end)) '%;']);
disp(['Test Accuracy: ' num2str(100*testAcc(end)) '%.']);
figure;
plot(0:epoch, 100*trainAcc);
hold on;
plot(0:epoch, 100*testAcc);
grid on;
title('Accuracy');
xlabel('Epoch');
ylabel('Accuracy (%)');
legend('Training', 'Test');
figure;
plot(0:epoch, trainLoss);
hold on;
plot(0:epoch, testLoss);
grid on;
title('Loss');
xlabel('Epoch');
ylabel(opt.loss);
legend('Training', 'Test');