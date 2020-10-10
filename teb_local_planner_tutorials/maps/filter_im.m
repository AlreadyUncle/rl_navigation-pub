
function im2 = filter_im(im)
    global WHITE;
    global BLACK;

    Radius = 1;
    size(im);
    im2 = im;
    for i = 1 : size(im)(1)
        for j = 1 : size(im)(2)
            if im(i,j) == BLACK && isolated(im,i,j,Radius)
                im2(i,j) = WHITE;
            end
        end
    end 

end

function t = isolated(im,i,j,Radius)
    global WHITE;
    global BLACK;

    row = size(im, 1);
    col = size(im, 2);
    xs = max(1 , i-Radius);
    ys = max(1 , j-Radius);
    xe = min(row, i+Radius);
    ye = min(col, j+Radius);
    ss = 0;
    for ii = xs : xe
        for jj = ys : ye
            if ii~= i && jj~=j
                if im(ii,jj) == BLACK
                    ss = ss + 1;
                    %return;
                end
            end
        end
    end
    
    if ss < 3
        t=1;
    else
        t=0;
    end

end