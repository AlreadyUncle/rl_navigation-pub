im = imread('lab3.pgm');

global WHITE;
global BLACK;

WHITE = 254;
BLACK = 0;


% im(200:250,280:350) = 255;
% im()

im_filtered = filter_im(im);
imshow(im_filtered);


% save pgm files
imwrite(im_filtered, 'lab3_filtered.pgm', 'pgm')


pause(200)