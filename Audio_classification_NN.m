load("MFCCdataset.mat");

%% Train Set
features = [];
features1 = [];
labels = [];
allLabels = adsTrain.Labels;
frameNum = 130;

for ii = 1:numel(mfccs)

    thismfcc = mfccs{ii};
    % Since each file has different length, pad 0 or truncate 
    % to keep same framelength(frameNum)
    if size(thismfcc,1) < frameNum
        thismfcc = padarray(thismfcc, (frameNum - size(thismfcc,1)), 0, 'post');
    elseif size(thismfcc,1) > frameNum
        thismfcc = thismfcc(1:frameNum,:);
        
    end
    label = allLabels(ii);
    
    features = [features,thismfcc];
    features1 = [features1;thismfcc];
    labels = [labels,label];
end

% Normalization and reshaping of dataset(Train)
M = mean(features1,1);
S = std(features1,[],1);

M1 = repmat(M,1,size(mfccs,1));
S1 = repmat(S,1,size(mfccs,1));

features = (features-M1)./S1;
features = reshape(features, frameNum, 13, 1, size(labels,2));


%% Test Set
features_ts = [];
labels_ts = [];

allLabels_ts = adsTest.Labels;

for ii = 1:numel(mfccs_ts)

    thismfcc = mfccs_ts{ii};
    numAxes = ndims(thismfcc);
    
    if size(thismfcc,1) < frameNum
        thismfcc = padarray(thismfcc, (frameNum - size(thismfcc,1)), 0, 'post');
    elseif size(thismfcc,1) > frameNum
        thismfcc = thismfcc(1:frameNum,:);
        
    end
    
    label = allLabels_ts(ii);

    features_ts = [features_ts,thismfcc];
    labels_ts = [labels_ts,label];
end

% Normalization and reshaping of dataset(Test)
M2 = repmat(M,1,size(mfccs_ts,1));
S2 = repmat(S,1,size(mfccs_ts,1));
features_ts = (features_ts-M2)./S2;
features_ts = reshape(features_ts, frameNum, 13,1, size(labels_ts,2));

%% Creating Layer
numAnimals = numel(unique(ads.Labels));

layers = [   

    imageInputLayer([frameNum 13 1])

    % Three convolutional layers
    
    convolution2dLayer(64,64,padding="same")
    batchNormalizationLayer
    leakyReluLayer(0.2)
    maxPooling2dLayer(2,Stride=2)
    
    convolution2dLayer(32,32,padding="same")
    batchNormalizationLayer
    leakyReluLayer(0.2)
    maxPooling2dLayer(2,Stride=2)
    
    convolution2dLayer(16, 16,padding="same")
    batchNormalizationLayer
    leakyReluLayer(0.2)
    maxPooling2dLayer(2,Stride=2)

    % This is followed by 3 fully-connected layers
    
    fullyConnectedLayer(256)
    batchNormalizationLayer
    leakyReluLayer(0.2)

    fullyConnectedLayer(256)
    batchNormalizationLayer
    leakyReluLayer(0.2)

    fullyConnectedLayer(256)
    batchNormalizationLayer
    leakyReluLayer(0.2)

    fullyConnectedLayer(numAnimals)
    softmaxLayer 
    classificationLayer];

%% Layer Summary
analyzeNetwork(layers)


%% Training and Testing Part

numEpochs = 20;
miniBatchSize = 100;
validationFrequency = floor(numel(labels)/miniBatchSize);

options = trainingOptions("sgdm", ...
    MiniBatchSize=miniBatchSize, ...
    Plots="training-progress", ...
    Verbose=true, ...
    MaxEpochs=numEpochs, ...
    ValidationData={features_ts,categorical(labels_ts)}, ...
    ValidationFrequency=validationFrequency);


[convNet,convNetInfo] = trainNetwork(features,labels,layers,options);

predictions = classify(convNet,features_ts);
predictions = categorical(string(predictions));
figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels_ts,predictions,title="Test Accuracy", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");
accuracy = sum(adsTest.Labels == predictions)/numel(adsTest.Labels)

%% Accuracy 
% 0.8057
