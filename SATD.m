%//*****************************************   
%//Copyright (c) 2015 Jingshuang Hu   
   
%//@filename:demo.m   
%//@datetime:2015.08.20   
%//@author:HJS   
%//@e-mail:eleftheria@163.com   
%//@blog:http://blog.csdn.net/hujingshuang   
%//*****************************************  
%% 
%//SATDģ��ƥ���㷨-����ķ�任(hadamard)
clear all;
close all;
%%
src=double(rgb2gray(imread('./datas/drone_satellite/scene_LPN1.jpg')));%//������ȵ�
mask=double(rgb2gray(imread('./datas/drone_satellite/template_LPN1.jpg')));%//������ȵ�
M=size(src,1);%//����ͼ��С
N=size(mask,1);%//ģ���С
%%
hdm_matrix=hadamard(N);%//hadamard�任����
hdm=zeros(M-N,M-N);%//����SATDֵ
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
title('ģ��');
figure;imshow(uint8(src));hold on;
rectangle('position',[y,x,N-1,N-1],'edgecolor','r');
title('�������');hold off;
%//��