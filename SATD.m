%//*****************************************   
%//Copyright (c) 2015 Jingshuang Hu   
   
%//@filename:demo.m   
%//@datetime:2015.08.20   
%//@author:HJS   
%//@e-mail:eleftheria@163.com   
%//@blog:http://blog.csdn.net/hujingshuang   
%//*****************************************  
%% 
%//SATD模板匹配算法-哈达姆变换(hadamard)
clear all;
close all;
%%
src=double(rgb2gray(imread('./datas/drone_satellite/scene_LPN1.jpg')));%//长宽相等的
mask=double(rgb2gray(imread('./datas/drone_satellite/template_LPN1.jpg')));%//长宽相等的
M=size(src,1);%//搜索图大小
N=size(mask,1);%//模板大小
%%
hdm_matrix=hadamard(N);%//hadamard变换矩阵
hdm=zeros(M-N,M-N);%//保存SATD值
for i=1:M-N
    for j=1:M-N
        temp=(src(i:i+N-1,j:j+N-1)-mask)/256;
        sw=(hdm_matrix*temp*hdm_matrix)/256;
        hdm(i,j)=sum(sum(abs(sw)));
    end
end
min_hdm=min(min(hdm));
[x y]=find(hdm==min_hdm);
figure;imshow(uint8(mask));
title('模板');
figure;imshow(uint8(src));hold on;
rectangle('position',[y,x,N-1,N-1],'edgecolor','r');
title('搜索结果');hold off;
%//完