//PARKPLATZKLASSE********************************************************************************************************************
/**
 * @version 0.02
 */
#include <string>
#include <math.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include "Parkplatz.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

Parkplatz::Parkplatz()
: corners(new Point2f[4]), ends(new Point[2]), filled(1), definedpoints(0), definedlength(0) {
}

Parkplatz Parkplatz::fromString(istringstream is, bool includeFilling) {
	Parkplatz p;
	for(int i = 0; i < 4; ++i) {
		Point2f* point = new Point2f();
		is >> point->x;
		is >> point->y;
		p.addPoint(point);
	}
	if(includeFilling) {
		bool filled;
		is >> filled;
		p.setFilled(filled);
	}
	return p;
}

Parkplatz* Parkplatz::fromString2(string str) {
	Parkplatz* res = new Parkplatz();
	vector<string> coordinates = ssplit(str, ';');
	if(coordinates.size() == 4) {
		for(int i = 0; i < 4; ++i) {
			vector<string> xy = ssplit(coordinates.at(i), ',');
			res->addPoint(new Point2f(atof(xy.at(0).c_str()), atof(xy.at(1).c_str())));
		}
	}
	return res;
}

vector<Parkplatz*> Parkplatz::getParkingsSpotsFromString(string str) {
	vector<Parkplatz*> res;
	vector<string> pSpotStrings = ssplit(str, ' ');
	for(int i = 0; i < pSpotStrings.size(); ++i) {
		res.push_back(Parkplatz::fromString2(pSpotStrings[i]));
	}
	return res;
}

int Parkplatz::addPoint(Point2f *p){
//	cout << "adding point " << p->x << ", " << p->y << endl;
	if(definedpoints<4){
		corners[definedpoints]=*p;
		definedpoints++;
	}
	else cout << "Parkplatz error: too many Points" << endl;

	//cout << definedpoints << definedpoints/4 << endl;
	return definedpoints/4;	//Return 1 when we need a new Parkplatz.
}

int Parkplatz::addLength(Point *p){
	if(definedlength<2){
		ends[definedlength]=*p;
		definedlength++;
	}
	else cout << "Length error: too many Points" << endl;
	
	//cout << definedpoints << definedpoints/4 << endl;
	return definedlength/2;	//Return 1 when we need a new Parkplatz.
}

string Parkplatz::getString(){
	stringstream result;
	for(int i = 0; i<definedpoints ; i++){
		result << corners[i].x << " " << corners[i].y << " ";
	}
	for(int i = 0; i<definedlength ; i++){
		result << ends[i].x << " " << ends[i].y << " ";
	}
	result << filled;
	return result.str();
}

string Parkplatz::getString2() {
	stringstream result;
	for(int i = 0; i < 3; ++i) {
		result << corners[i].x << "," << corners[i].y << ";";
	}
	result << corners[3].x << "," << corners[3].y;
	return result.str();
}

Point2f Parkplatz::getCenter(){
	double cx = 0,cy =0;
	for(int i = 0; i<4 ; i++)
	{
		cx+=corners[i].x/4;
		cy+=corners[i].y/4;
	}
	Point2f center(cx-abs(corners[0].x-cx)*0.5,cy);
	return center;
}

bool Parkplatz::isFilled(){
	return filled;
}

void Parkplatz::setFilled(bool filled) {
	this->filled = filled;
}

void Parkplatz::getBoundary(Mat* res) {
	vector<Point2f> points;
	points.push_back(corners[0]);
	points.push_back(corners[1]);
	points.push_back(corners[2]);
	points.push_back(corners[3]);
	Mat mat(points);
	mat.copyTo(*res);
}

void Parkplatz::getCorners(Point* corners) {
	for(int i=0;i<4;++i)
		corners[i] = Point((int)this->corners[i].x, (int)this->corners[i].y);
}

void Parkplatz::editCorners(Point* incorners){
    for(int i=0;i<4;++i){
        corners[i].y=incorners[i].y;
    }
}
