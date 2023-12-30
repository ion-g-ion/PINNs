
R = 1.0;
r = 0.2;

Point(1) = {0,R,0,meshsize};
Point(2) = {R,0,0,meshsize};
Point(3) = {R/1.41421356237,R/1.41421356237,0,meshsize};
Point(11) = {0.0,0.0,0,meshsize};
Point(12) = {0,r,0,meshsize/5};
Point(13) = {r,0,0,meshsize/5};
Point(14) = {r,r,0,meshsize/5};//+
Line(1) = {12, 14};
//+
Line(2) = {11, 13};
//+
Line(3) = {13, 14};
//+
Line(4) = {14, 3};
//+
Line(5) = {12, 11};
//+
Line(6) = {13, 2};
//+
Line(7) = {1, 12};
//+
Circle(8) = {1, 11, 3};
//+
Circle(9) = {3, 11, 2};
//+
Physical Curve(10) = {7, 5, 2, 6, 9, 8};
//+
Curve Loop(1) = {1, -3, -2, -5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {7, 1, 4, -8};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {4, 9, -6, 3};
//+
Plane Surface(3) = {3};
//+
Physical Surface(11) = {1};
//+
Physical Surface(12) = {2, 3};
