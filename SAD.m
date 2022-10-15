%%
%���������㷨��SAD��
clear all;
close all;
%%
root_dir = './datas/drone_satellite/';
src=imread('./datas/drone_satellite/scene_LPN1.jpg');
[a b d]=size(src);
if d==3
    src=rgb2gray(src);
end

mask=imread('./datas/drone_satellite/template_LPN1.jpg');
[m n d]=size(mask);
if d==3
    mask=rgb2gray(mask);
end
%%
N=n;%ģ��ߴ磬Ĭ��ģ��Ϊ������
M=a;%������ͼ��ߴ磬Ĭ������ͼ��Ϊ������
%%
dst=zeros(M-N,M-N);
for i=1:M-N         %��ͼѡȡ��ÿ�λ���һ������
    for j=1:M-N
        temp=src(i:i+N-1,j:j+N-1);%��ǰ��ͼ
        dst(i,j)=dst(i,j)+sum(sum(abs(temp-mask)));
    end
end
abs_min=min(min(dst));
[x,y]=find(dst==abs_min);
figure;
imshow(mask);title('ģ��');
figure;
imshow(src);
hold on;
rectangle('position',[y,x,N-1,N-1],'edgecolor','r');
hold off;title('����ͼ');