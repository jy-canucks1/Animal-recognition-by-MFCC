path = "C:\Users\User\Desktop\Winter 2024\URA\Animal-Sound-Dataset-master"


dataFolder = fullfile(path + "\original_dataset")

ads = audioDatastore(dataFolder,IncludeSubfolders=true,LabelSource="foldernames");

% Create a new AudioDatastore with a custom transform function

labelTable = countEachLabel(ads)

classes = labelTable.Label;
numClasses = size(labelTable,1);

[adsTrain,adsTest] = splitEachLabel(ads,0.8,0.2);

[sampleTrain,dsInfo] = read(adsTrain);
%sound(sampleTrain,dsInfo.SampleRate)

fs = dsInfo.SampleRate;
windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);
afe = audioFeatureExtractor(SampleRate=fs, ...
    Window=hamming(windowLength,"periodic"),OverlapLength=overlapLength, ...
    zerocrossrate=false,shortTimeEnergy=false,pitch=false,mfcc=true);
setExtractorParams(afe,"mfcc","NumCoeffs",13)
mfccs = extract(afe,adsTrain,SampleRateMismatchRule="resample",UseParallel=true);
mfccs_ts = extract(afe,adsTest,SampleRateMismatchRule="resample",UseParallel=true);

save MFCCdataset.mat;