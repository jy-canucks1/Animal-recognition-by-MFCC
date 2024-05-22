path = "C:\Users\User\Desktop\Winter 2024\URA\Animal-Sound-Dataset-master"

% Specify the main folder containing subfolders with mixed audio files
mainFolder = path + "\original_dataset";
% MFCC of some of audio samples 
tiledlayout(4,2)

[audioIn1,fs1] = audioread(mainFolder +"/Cat/cat_6.wav");
[coeffs1,delta1,deltaDelta1,loc1] = mfcc(audioIn1,fs1);


[audioIn2,fs2] = audioread(mainFolder +"/Cat/cat_2.wav");
[coeffs2,delta2,deltaDelta2,loc2] = mfcc(audioIn2,fs2);


[audioIn3,fs3] = audioread(mainFolder +"/Lion/aslan_4.wav");
[coeffs3,delta3,deltaDelta3,loc3] = mfcc(audioIn3,fs3);

[audioIn4,fs4] = audioread(mainFolder +"/Lion/aslan_10_mono.wav");
[coeffs4,delta4,deltaDelta4,loc4] = mfcc(audioIn4,fs4);

[audioIn5,fs5] = audioread(mainFolder +"/Dog/dog_35.wav");
[coeffs5,delta5,deltaDelta5,loc5] = mfcc(audioIn5,fs5);


[audioIn6,fs6] = audioread(mainFolder +"/Dog/dog_140.wav");
[coeffs6,delta6,deltaDelta6,loc6] = mfcc(audioIn6,fs6);

[audioIn7,fs7] = audioread(mainFolder +"/Chicken/tavuk_3.wav");
[coeffs7,delta7,deltaDelta7,loc7] = mfcc(audioIn7,fs7);

[audioIn8,fs8] = audioread(mainFolder +"/Chicken/tavuk_30.wav");
[coeffs8,delta8,deltaDelta8,loc8] = mfcc(audioIn6,fs6);


nexttile
mfcc(audioIn1,fs1);
nexttile
mfcc(audioIn2,fs2);
nexttile
mfcc(audioIn3,fs3);
nexttile
mfcc(audioIn4,fs4);
nexttile
mfcc(audioIn5,fs5);
nexttile
mfcc(audioIn6,fs6);
nexttile
mfcc(audioIn7,fs7);
nexttile
mfcc(audioIn8,fs8);


dataFolder0 = dir(mainFolder + "\*\*.wav")
length(dataFolder0)

for i= 1: length(dataFolder0)
    path1 = strcat(dataFolder0(i).folder,'\',dataFolder0(i).name);
    [audioIn, fs]=audioread(path1);
    if size(audioIn, 2) == 2
    % Convert stereo to mono by averaging channels
    audioIn = mean(audioIn, 2);
    audiowrite([path1(1 : end - 4) , '_mono.wav'] , audioIn , fs)
    end
end

dataFolder = fullfile(mainFolder)
        
ads = audioDatastore(dataFolder,IncludeSubfolders=true,LabelSource="foldernames");
% stereo to mono
disp(ads)
% Create a new AudioDatastore with a custom transform function
labelTable = countEachLabel(ads)

classes = labelTable.Label;
numClasses = size(labelTable,1);

[adsTrain,adsTest] = splitEachLabel(ads,0.8,0.2);

[sampleTrain,dsInfo] = read(adsTrain);

fs = dsInfo.SampleRate;

windowLength = round(0.03*fs);
overlapLength = round(0.025*fs);
afe = audioFeatureExtractor(SampleRate=fs, ...
    Window=hamming(windowLength,"periodic"),OverlapLength=overlapLength, ...
    zerocrossrate=false,shortTimeEnergy=false,pitch=false,mfcc=true);
setExtractorParams(afe,"mfcc","NumCoeffs",13)
featureMap = info(afe)
afe
disp(afe)

allFeatures = extract(afe,adsTrain,SampleRateMismatchRule="resample",UseParallel=true);
dimensions = size(allFeatures)
mfccs = allFeatures(:,:);

size(mfccs)