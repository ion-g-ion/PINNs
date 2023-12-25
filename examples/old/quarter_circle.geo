
//meshsize = 0.025;

R = 2;
r = 1;

Point(0) = {R,0,0,meshsize};
Point(1) = {r,0,0,meshsize};
Point(2) = {0,r,0,meshsize};
Point(3) = {0,R,0,meshsize};
Point(4) = {0,0,0,1};//+
Line(1) = {1, 0};
//+
Line(2) = {2, 3};
//+
Circle(3) = {2, 4, 1};
//+
Circle(4) = {3, 4, 0};
//+
Curve Loop(1) = {4, -1, -3, 2};
//+
Plane Surface(1) = {1};
//+
Physical Curve("not_fixed", 5) = {4, 1, 3};
//+
Physical Curve("fixed", 6) = {2};
//+
Physical Surface(7) = {1};
