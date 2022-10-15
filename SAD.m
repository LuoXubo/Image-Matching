%%
%绝对误差和算法（SAD）
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
N=n;%模板尺寸，默认模板为正方形
M=a;%代搜索图像尺寸，默认搜索图像为正方形
%%
dst=zeros(M-N,M-N);
for i=1:M-N         %子图选取，每次滑动一个像素
    for j=1:M-N
        temp=src(i:i+N-1,j:j+N-1);%当前子图
        dst(i,j)=dst(i,j)+sum(sum(abs(temp-mask)));
    end
end
abs_min=min(min(dst));
[x,y]=find(dst==abs_min);
figure;
imshow(mask);title('模板');
figure;
imshow(src);
hold on;
rectangle('position',[y,x,N-1,N-1],'edgecolor','r');
hold off;title('搜索图');