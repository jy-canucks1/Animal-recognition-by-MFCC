load("MFCCdataset.mat");

features = [];
labels = [];
allLabels = adsTrain.Labels;
frameNum = 130;

for ii = 1:numel(mfccs)

    thismfcc = mfccs{ii};
    if size(thismfcc,1) < frameNum
        thismfcc = padarray(thismfcc, (frameNum - size(thismfcc,1)), 0, 'post');
    elseif size(thismfcc,1) > frameNum
        thismfcc = thismfcc(1:frameNum,:);
        
    end
    label = repelem(allLabels(ii),size(thismfcc,1));
    features = [features;thismfcc];
    labels = [labels,label];
end


M = mean(features,1);
S = std(features,[],1);
features = (features-M)./S;


trainedClassifier = fitcensemble(array2table(features),labels,'Method','AdaBoostM2');

k = 5;
group = labels;
c = cvpartition(group,KFold=k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,CVPartition=c);

validationPredictions = kfoldPredict(partitionedModel);
figure(Units="normalized",Position=[0.4 0.4 0.4 0.4])
confusionchart(labels,validationPredictions,title="Validation Accuracy", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");

features_ts = [];
labels_ts = [];
numVectorsPerFile = [];


allLabels_ts = adsTest.Labels;

for ii = 1:numel(mfccs_ts)

    thismfcc = mfccs_ts{ii};

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

