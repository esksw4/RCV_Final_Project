%%shorter version to save time
% run('/Users/e.kim4/Downloads/vlfeat-0.9.20/toolbox/vl_setup')
% vl_version verbose
%% 1: Bag Of Words
clear;
close all;
clc;

imgDir = '/Users/e.kim4/Documents/MATLAB/matconvnet-1.0-beta23/class';
imds = imageDatastore(imgDir, 'IncludeSubfolders', true,...
       'LabelSource', 'foldernames');
%split into testing vs train images: split is 10,10
%%Project4 -> Project5: added so it can take diff num of test and training
%%images
numTestImgClass = 10; 
[testImages,trainImages] = splitEachLabel(imds,10);
%label the trainImages 1~10
trainlabels = grp2idx(trainImages.Labels);
sizelabels = size(trainlabels);
concatDesTrain = [];

for i = 1:5%sizelabels(1)
    train{i} = imread(trainImages.Files{i});
    test{i} = imread(testImages.Files{i});
    %read each train images & test images
    if size(train{i},3) ~= 3
        train{i} = imresize(train{i},[300,300]);
    elseif size(train{i},3) ~= 3
        test{i} = imresize(test{i},[300,300]);
    else
        train{i} = rgb2gray(imresize(train{i},[300,300]));
        test{i} = rgb2gray(imresize(test{i},[300,300]));
    end

    %format them as single (just to run it on vl_sift)
    %singleTrain{i} = single(train{i});
    %singleTest{i} = single(test{i});
    
    %For a subset of all the interstpoints in training image, clustering
    %the descriptors using k-means clustering:
    %output as k-visual words with each word has an associated 128x1
    %centroids 
    %%%USING VL_SIFT: find interest points and descriptor for testing and
    %%%train images
    %[interestTrain{i}, desTrain{i}] = vl_sift(singleTrain{i});
    %[interestTest{i}, desTest{i}] = vl_sift(singleTest{i});
    
    %USING MATLAB BUILT IN FUNCTION
    interestTrain{i} = detectSURFFeatures(train{i});
    %getOnly 150 strongest interestpoint to compute
    interestTrain{i} = interestTrain{i}.selectStrongest(100);
    [desTrain{i}, validPTrain{i}] = extractFeatures(train{i},...
                                    interestTrain{i},'SURFSize',128);
    
    interestTest{i} = detectSURFFeatures(test{i});
    interestTest{i} = interestTest{i}.selectStrongest(100);
    [desTest{i}, validPTest{i}] = extractFeatures(test{i},...
                                  interestTest{i},'SURFSize',128);
    
    %put all the descriptors in one matrix for testing and train images
    concatDesTrain = [concatDesTrain desTrain{i}'];
    %concatDesTest = [concatDesTest desTest{i}];    
end;

K = 300;
%so in here, assignments = closest centroids for each of Descriptor.
[centroid, assignTrain] = vl_kmeans(double(concatDesTrain), K); 

%seperate each assigned descriptors into each classes.
for i = 1:5%%sizelabels(1)
   [closestToCentroidsTrain{i}, distanceTrain{i}] = knnsearch(centroid',...
                                                    desTrain{i});
end

for i = 1:5%%sizelabels(1)
   [closestToCentroidsTest{i}, distanceTest{i}] = knnsearch(centroid',...
                                                  desTest{i});
end

for i = 1:5%%sizelabels(1)
    % each image is represented with 300x1 and there are 50 images
    featureVectorTrain(:,i) = hist(closestToCentroidsTrain{i},K);
    featureVectorTest(:,i) = hist(closestToCentroidsTest{i},K);
end

for i = 1:5%%sizelabels(1)
   [concatFVTestLabel(i,1), concatFVTestDistance(i,1)] = knnsearch(...
                              featureVectorTrain',featureVectorTest(:,i)');
end

n = 1; %keep track of the ranges.
%%keep track on class number
    for i = 1:5%%size(concatFVTestLabel,1)
        if concatFVTestLabel(i) >= n &&...
           concatFVTestLabel(i) < n+numTestImgClass
                concatFVTestLabel(i) = 1;
        elseif concatFVTestLabel(i) >= n+numTestImgClass...
               && concatFVTestLabel(i) < n+(2*numTestImgClass)
                    concatFVTestLabel(i) = 2;
        elseif concatFVTestLabel(i) >= n+(2*numTestImgClass) ...
               && concatFVTestLabel(i) < n+(3*numTestImgClass)
                    concatFVTestLabel(i) = 3;
        elseif concatFVTestLabel(i) >= n+(3*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(4*numTestImgClass)
                    concatFVTestLabel(i) = 4;
        elseif concatFVTestLabel(i) >= n+(4*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(5*numTestImgClass)
                    concatFVTestLabel(i) = 5;
        elseif concatFVTestLabel(i) >= n+(5*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(6*numTestImgClass)
                    concatFVTestLabel(i) = 6;
        elseif concatFVTestLabel(i) >= n+(6*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(7*numTestImgClass)
                    concatFVTestLabel(i) = 7;
        elseif concatFVTestLabel(i) >= n+(7*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(8*numTestImgClass)
                    concatFVTestLabel(i) = 8;
        elseif concatFVTestLabel(i) >= n+(8*numTestImgClass)... 
               && concatFVTestLabel(i) < n+(9*numTestImgClass)
                    concatFVTestLabel(i) = 9;
        else
            concatFVTestLabel(i) = 10;
        end
    end
    
 %% show similar histograms  
%%show different histogram figure(2)
% since I'm only showing 1 image I commented out others
figure(1) 
i = 0;
            subplot(3,2,1);
            hist(closestToCentroidsTrain{1+i},K)
            %featureVectorTest{trackTrain} = dummyTrain.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain(:,1+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(1+i),trainImages.Labels(1+i))); hold off;
          
            i = 4;
            subplot(3,2,2);
            hist(closestToCentroidsTrain{1+i},K)
            %featureVectorTest{trackTest} = dummyTest.Values; 
            hold on;
            xlabel('K')
            ylabel('frequency')
            axis([0 K+1 0  max(featureVectorTrain(:,1+i))]);
            title(sprintf('Differneces: train class %d\n%s',trainlabels(1+i),trainImages.Labels(1+i))); hold off;
%             
%             subplot(3,2,3);
%             hist(closestToCentroidsTrain{80+i},K)
%             %featureVectorTest{trackTrain} = dummyTrain.Values; 
%             hold on;
%             xlabel('K')
%             ylabel('frequency')
%             axis([0 K+1 0  max(featureVectorTrain(:,80+i))]);
%             title(sprintf('Differneces: train class %d\n%s',trainlabels(80+i),trainImages.Labels(80+i))); hold off;
%             
%             i = 2;
%             subplot(3,2,4);
%             hist(closestToCentroidsTrain{80+i},K)
%             %featureVectorTest{trackTest} = dummyTest.Values; 
%             hold on;
%             xlabel('K')
%             ylabel('frequency')
%             axis([0 K+1 0  max(featureVectorTrain(:,80+i))]);
%             title(sprintf('Differneces: train class %d\n%s',trainlabels(80+i),trainImages.Labels(80+i))); hold off;
%             
%             subplot(3,2,5);
%             hist(closestToCentroidsTrain{40+i},K)
%             %featureVectorTest{trackTrain} = dummyTrain.Values; 
%             hold on;
%             xlabel('K')
%             ylabel('frequency')
%             axis([0 K+1 0  max(featureVectorTrain(:,40+i))]);
%             title(sprintf('Differneces: train class %d\n%s',trainlabels(40+i),trainImages.Labels(40+i))); hold off;
%             
%             i = 7;
%             subplot(3,2,6);
%             hist(closestToCentroidsTrain{40+i},K)
%             %featureVectorTest{trackTest} = dummyTest.Values; 
%             hold on;
%             xlabel('K')
%             ylabel('frequency')
%             axis([0 K+1 0  max(featureVectorTrain(:,40+i))]);
%             title(sprintf('Differneces: train class %d\n%s',trainlabels(40+i),trainImages.Labels(40+i))); hold off;
%confusionMat doesn't work because there is only one class
%stats = confusionmatStats(trainlabels(1:5),concatFVTestLabel);