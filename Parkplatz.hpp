//PARKPLATZKLASSE********************************************************************************************************************
/**
 * @version 0.02
 */
#ifndef PARKPLATZ_H_
#define PARKPLATZ_H_

#include <vector>
#include <opencv/cv.h>

using namespace cv;
using namespace std;

class Parkplatz {
public:
    Parkplatz();
    Parkplatz(Point2f* corners);
	int addPoint(Point2f *p);
    int addLength(Point *p);
	string getString();
	string getString2();
    Point2f getCenter();
    bool isFilled();
    void setFilled(bool filled);
    void editCorners(Point* corners);
    void getBoundary(Mat* res);
    void getCorners(Point* corners);
	Point2f* corners;
    Point* ends;

    static Parkplatz fromString(istringstream is, bool includeFilling);
    static Parkplatz* fromString2(string str);
    static vector<Parkplatz*> getParkingsSpotsFromString(string str);
private:
	bool filled;
	int definedpoints;
    int definedlength;
};

#endif
