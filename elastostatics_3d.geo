//+
// SetFactory("OpenCASCADE");


// meshsize = 0.15;

R = 2;
r = 1;
Rp = 2;
d = 0.5;

Point(0) = {r,0,0,meshsize/1};
Point(1) = {0,0,0,meshsize};
Point(2) = {0,0,r,meshsize/2};
Point(3) = {r,d,0,meshsize/1};
Point(4) = {0,d,0,meshsize};
Point(5) = {0,d,r,meshsize/2};
Point(6) = {R,0,0,meshsize/1};
Point(7) = {0,0,R,meshsize/2};
Point(8) = {R,d,0,meshsize/1};
Point(9) = {0,d,R,meshsize/2};//+

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



Point(30) = {r+Rp,-r,R,meshsize};
Point(31) = {r+Rp,-r,r,meshsize};
Point(32) = {r+Rp,-R,R,meshsize};
Point(33) = {r+Rp,-R,r,meshsize};

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
Line(111) = {7,9};//+


//+
Ellipse(112) = {9, 14, 9, 19};
//+
Ellipse(113) = {18, 14, 18, 16};
//+
Ellipse(114) = {5, 11, 5, 13};
//+
Ellipse(115) = {12, 11, 12, 10};
//+
Line(116) = {18, 9};
//+
Line(117) = {12, 5};
//+
Line(118) = {18, 12};
//+
Line(119) = {12, 22};
//+
Line(120) = {22, 2};
//+
Line(121) = {7, 23};
//+
Line(122) = {18, 23};
//+
Line(123) = {23, 22};
//+
Line(124) = {16, 19};
//+
Line(125) = {13, 19};
//+
Line(126) = {16, 10};
//+
Line(127) = {10, 13};
//+
Line(128) = {30, 32};
//+
Line(129) = {32, 33};
//+
Line(130) = {33, 31};
//+
Line(131) = {31, 30};

//+
Ellipse(132) = {7, 21, 7, 30};
//+
Ellipse(133) = {2, 20, 2, 31};
//+
Ellipse(134) = {22, 20, 22, 33};

//+
Ellipse(135) = {23, 21, 23, 32};
//+
Curve Loop(1) = {111, -116, 122, -121};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {116, -106, -117, -118};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {123, -119, -118, 122};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {120, 104, 121, 123};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {109, -117, 119, 120};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {109, 106, -111, -104};
//+
Plane Surface(6) = {6};
//+
//+
Surface Loop(1) = {2, 1, 6, 5, 3, 4};
//+
Volume(1) = {1};
//+
Curve Loop(7) = {107, -110, -105, 108};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {101, 106, -103, -107};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {102, -104, -100, 105};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {102, 111, -103, -110};
//+
Surface(10) = {10};
//+
Curve Loop(12) = {101, -109, -100, 108};
//+
Surface(11) = {12};
//+
Surface Loop(2) = {9, 10, 8, 11, 7, 6};
//+
Volume(2) = {2};
//+
Curve Loop(14) = {133, -130, -134, 120};
//+
Plane Surface(12) = {14};
//+
Curve Loop(15) = {114, -127, -115, 117};
//+
Plane Surface(13) = {15};
//+
//+
//+
Extrude {0, 0, 1} {
  Surface{12}; 
}
//+
Extrude {0, 0, 1} {
  Surface{13}; 
}
//+
Physical Surface("D", 152) = {7};
//+
Physical Volume("V1", 1001) = {2};
//+
Physical Volume("V2", 1002) = {1};
//+
Physical Volume("V3", 1003) = {3};
//+
Physical Volume("V4", 1004) = {4};
//+
// Physical Volume(1005) = {4, 1, 2, 3};
//+
Physical Surface(153) = {9, 11, 10, 8, 19, 23, 21, 13, 20, 15, 16, 14, 12, 18, 1, 3, 5};
