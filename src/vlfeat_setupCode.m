%%
run('/Users/e.kim4/Downloads/vlfeat-0.9.20/toolbox/vl_setup')
vl_version verbose
%%
untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta23.tar.gz') ;
cd matconvnet-1.0-beta23
run matlab/vl_compilenn ;
%%
urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
  'imagenet-vgg-f.mat') ;
run matlab/vl_setupnn ;
%%
net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;
im = imread('peppers.png') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = im_ - net.meta.normalization.averageImage ;

% Run the CNN.
res = vl_simplenn(net, im_) ;
%%
% Show the classification result.
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
   net.meta.classes.description{best}, best, bestScore)) ;
