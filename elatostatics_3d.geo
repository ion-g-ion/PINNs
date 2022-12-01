//+
SetFactory("OpenCASCADE");


meshsize = 0.1;

R = 2;
r = 1;
Rp = 2;
d = 0.5;

Point(0) = {r,0,0,meshsize};
Point(1) = {0,0,0,meshsize};
Point(2) = {0,0,r,meshsize};
Point(3) = {r,d,0,meshsize};
Point(4) = {0,d,0,meshsize};
Point(5) = {0,d,r,meshsize};
Point(6) = {R,0,0,meshsize};
Point(7) = {0,0,R,meshsize};
Point(8) = {R,d,0,meshsize};
Point(9) = {0,d,R,meshsize};//+

Point(10) = {r+Rp,R+d,r,meshsize};
Point(12) = {-(R-r),d,r,meshsize};
Point(13) = {r+Rp,r+d,r,meshsize};
Point(16) = {r+Rp,R+d,R,meshsize};
Point(18) = {-(R-r),d,R,meshsize};
Point(19) = {r+Rp,r+d,R,meshsize};
Point(11) = {r+Rp,d,r,meshsize};
Point(14) = {r+Rp,d,R,meshsize};

Point(20) = {r+Rp,0,r,meshsize};
Point(21) = {r+Rp,0,R,meshsize};
Point(22) = {-r,0,r,meshsize};
Point(23) = {-r,0,R,meshsize};
Point(20) = {r+Rp,0,r,meshsize};
Point(21) = {r+Rp,0,R,meshsize};
Point(11) = {r+Rp,d,r,meshsize};
Point(14) = {r+Rp,d,R,meshsize};

Circle(100) = {0,1,2};
Circle(101) = {3,4,5};
Circle(102) = {6,1,7};
Circle(103) = {8,4,9};
Line(104) = {2,7};
Line(105) = {0,6};
Line(106) = {5,9};
Line(107) = {3,8};
Line(108) = {0,3};
Line(109) = {2,5};
Line(110) = {6,8};
Line(111) = {7,9};