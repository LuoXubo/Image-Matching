unregistered = imread('./datas/drone_satellite/template_LPN1.jpg');
scene = imread('./datas/drone_satellite/scene_LPN1.jpg');
% imshow(unregistered);

[mp, fp] = cpselect(unregistered, scene, 'Wait', true);
t = fitgeotrans(mp,fp,"projective");

Rfixed = imref2d(size(scene));
registered = imwarp(unregistered,t);
imshow(registered);

%imshowpair(ortho,registered,"blend");

