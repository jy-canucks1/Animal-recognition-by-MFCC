load("MFCCdataset.mat");

%% Training and Validation Part
features = [];
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
    % Label each frame
    label = repelem(allLabels(ii),size(thismfcc,1));
    
    features = [features;thismfcc];
    labels = [labels,label];
end

% Normalization and reshaping of dataset(Train)
M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;


trainedClassifier = fitcknn(features,labels, ...
    Distance="euclidean", ...
    NumNeighbors=15, ...
    DistanceWeight="squaredinverse", ...
    Standardize=false, ...
    ClassNames=unique(labels));

k = 5;
group = labels;
c = cvpartition(group,KFold=k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,CVPartition=c);

validationPredictions = kfoldPredict(partitionedModel);
figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels,validationPredictions,title="Validation Accuracy", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");

%% Testing Part
features_ts = [];
labels_ts = [];
numVectorsPerFile = [];


allLabels_ts = adsTest.Labels;

for ii = 1:numel(mfccs_ts)

    thismfcc = mfccs_ts{ii};
    numAxes = ndims(thismfcc);
    
    if size(thismfcc,1) < frameNum
        thismfcc = padarray(thismfcc, (frameNum - size(thismfcc,1)), 0, 'post');
    elseif size(thismfcc,1) > frameNum
        thismfcc = thismfcc(1:frameNum,:);
        
    end
    
    numVec = size(thismfcc,1);
    label = repelem(allLabels_ts(ii),numVec);
    
    numVectorsPerFile = [numVectorsPerFile,numVec];
    features_ts = [features_ts;thismfcc];
    labels_ts = [labels_ts,label];
end

% Normalization and reshaping of dataset(Test)
features_ts = (features_ts-M)./S;

prediction = predict(trainedClassifier,features_ts);
prediction = categorical(string(prediction));

figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels_ts(:),prediction,title="Test Accuracy (Per Frame)", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");

r2 = prediction(1:numel(adsTest.Files));

idx = 1;
for ii = 1:numel(adsTest.Files)
    thismfcc = mfccs_ts{ii};
    r2(ii) = mode(prediction(idx:idx+numVectorsPerFile(ii)-1));
    idx = idx + numVectorsPerFile(ii);
end

figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(adsTest.Labels,r2,title="Test Accuracy (Per File)", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");

accuracy = sum(adsTest.Labels == r2)/numel(adsTest.Labels)
%% Accuracy (k: NumNeighbors)
% k=5 -> 0.8571
% k=7 -> 0.8571
% k=9 -> 0.8629
% k=11 -> 0.8514
% k=13 -> 0.8571
% k=15 -> 0.8571