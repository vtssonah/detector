//
//  image_alg.cpp
//  
//
//  Created by Victor ter Smitten on 30/10/16.
//
//
#include <limits>
#include <string>
#include <iostream>
#include "opencv/cv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "image_alg.hpp"

using namespace std;
using namespace cv;

	//**************************************************************************************
    //calculates reldev without mask
double calreldev(Mat frame){
    Scalar mean;
    Scalar stdev;
    meanStdDev(frame,mean,stdev);
    double  reldev=(pow(stdev.val[0],2)/(pow(mean.val[0]/255,2)+1));
    if(reldev==0){
        cout<<"Screen is single colored, Please Check!!!!!!!!!!!!!!!_whole"<< stdev.val[0] << endl;
        reldev=1;
    }
    return reldev;
}
    //calculates reldev with mask
double calreldev(Mat frame, Mat mask){
    Scalar mean;
    Scalar stdev;
    meanStdDev(frame,mean,stdev,mask);
    double  reldev=(pow(stdev.val[0],2)/(pow(mean.val[0]/255,2)+1));
    if(reldev==0){
            //cout<<stdev.val[0]<<"std"<<", mean:"<<mean.val[0]<<endl;
        cout<<"Screen is single colored, Please Check!!!!!!!!!!!!!!!_mask"<< stdev.val[0]<<endl << "Problem could originate from using release libraries in debug mode!" << endl;
        reldev=1;
    }
    return reldev;
}
    //calculates carlength of the to calibration Points
int getcarlength(Parkplatz* spot,bool& leftright){
    int carside1x=1;//spot->ends[0].x;
    int carside1y=1;//spot->ends[0].y;
    int carside2x=1;//spot->ends[1].x;
    int carside2y=1;//spot->ends[1].y;
    int carlength = sqrt(pow((carside1x-carside2x),2)+pow((carside1y-carside2y),2));
    if(abs(carside1x-carside2x)>abs(carside1y-carside2y)){
        leftright=true;
    }else{leftright=false;}
        //cout<<"******************************************************** carlength "<<carlength<<endl;
        return carlength;
}

bool my_comp(const Point& a, const Point& b)
{
    return sqrt(pow(a.x,2)+pow(a.y,2))<sqrt(pow(b.x,2)+pow(b.y,2));
}

    //**************************************************************************************
    //Sorts a vector of 4 points into top left, bottomleft, bottomright, top right
vector<Point> sortPoints(vector<Point> unsorted){
    vector<Point> sorted;
        //cout<<unsorted<<" start sort"<<endl;
    for (int i = 0; i < 4; i++){(sorted.push_back(Point(0,0)));}
    int x=0;
    int a=unsorted[0].x;
    int b=unsorted[0].y;
    for (int i= 1;i<unsorted.size();i++){
        if (sqrt(pow(unsorted[i].x,2)+pow(unsorted[i].y,2))<sqrt(pow(a,2)+pow(b,2))){
            a=unsorted[i].x;
            b=unsorted[i].y;
            x=i;
                //cout<<a<<" "<<b<<" "<<x<<" "<<endl;
        }
    }
    sorted[0]=unsorted.at(x);
    x=x+1;
    if(x>3){
        x=x-4;
    }
    sorted[1]=unsorted.at(x);
    x=x+1;
    if(x>3){
        x=x-4;
    }
    sorted[2]=unsorted.at(x);
    x=x+1;
    if(x>3){
        x=x-4;
    }
    sorted[3]=unsorted.at(x);
    if (sorted[0].y>sorted[2].y && sorted[1].y>sorted[3].y){//&& sorted[1].y<sorted[3].y
        a=sorted[0].x;
        b=sorted[0].y;
        sorted[0].x=sorted.at(3).x;
        sorted[0].y=sorted.at(3).y;
        sorted[3].x=sorted.at(2).x;
        sorted[3].y=sorted.at(2).y;
        sorted[2].x=sorted.at(1).x;
        sorted[2].y=sorted.at(1).y;
        sorted[1].x=a;
        sorted[1].y=b;
        
    }
    if (sorted[1].x>sorted[3].x){//&& sorted[1].y<sorted[3].y
        a=sorted[1].x;
        b=sorted[1].y;
        sorted[1].x=sorted.at(3).x;
        sorted[1].y=sorted.at(3).y;
        sorted[3].x=a;
        sorted[3].y=b;
    }
        //cout<<sorted<<" ausgang sorted"<<endl;
    return sorted;
}
	//**************************************************************************************
    //Parkfeld aufteilen
vector<Point*> split_pf(vector<Point> pfield,int factor, bool leftright, float& frac_mid,bool warped){
    vector<Point*> mops_strips;
        //calculate gradients
        //cout<<"warped:"<<warped<<"leftright:"<<leftright<<"pfield:"<<pfield<<endl;
    double m_bottom_nom=pfield[2].x-pfield[1].x;
    double m_top_nom=pfield[3].x-pfield[0].x;

    double m_bottom = double((pfield[2].y-pfield[1].y)/m_bottom_nom);
    double m_top = double((pfield[3].y-pfield[0].y)/m_top_nom);
    if(m_top_nom==0){
        m_top=0;}
    if(m_bottom_nom==0){
        m_bottom=0;}
    int div,div2;
        //cout<<leftright<<" leftright1 "<<pfield<<endl;
    if (leftright){
        div=abs((abs(pfield[0].x-pfield[3].x)+abs(pfield[1].x-pfield[2].x))/2)/factor;
            //cout<<div<<" div"<<endl;
    }
    else{
        div=abs((abs(pfield[0].y-pfield[3].y)+abs(pfield[1].y-pfield[2].y))/2)/factor;
            //cout<<div<<" div0"<<endl;
    }
    float frac_bottom,frac_top,frac_bottom2,frac_top2;
    frac_top2=0;
    frac_bottom2=0;
    
    frac_bottom=(pfield[2].x-pfield[1].x)/double(div);
    frac_top=(pfield[3].x-pfield[0].x)/double(div);
    frac_mid=(frac_bottom+frac_top)/2;
        //cout<<"lr"<<leftright<<"frac_mid"<<frac_mid<<endl;
    
    if(leftright==0){
        div=abs((abs(pfield[2].y-pfield[1].y)+abs(pfield[3].y-pfield[0].y))/2)/factor;
        frac_bottom2=abs((pfield[2].y-pfield[1].y)/double(div));
        frac_top2=abs((pfield[3].y-pfield[0].y)/double(div));
        frac_mid=(frac_bottom2+frac_top2)/2;
        m_bottom=0;
        m_top=0;
            //cout<<div<<" div0"<<endl;
    }
        //cout<<"lr"<<leftright<<"frac_mid"<<frac_mid<<endl;
        //cout<<"div:"<<div<<" frac_bottom:"<<frac_bottom<<endl;
    for (int i=1;i<div;i++){
        
            //linke Ecke festlegen
        if (i==1){
            Point* corners;
            corners=new Point[4];
            corners[1]=Point(pfield[1].x,pfield[1].y);
            corners[2]=Point((pfield[1].x+frac_bottom),(m_bottom*frac_bottom+pfield[1].y+frac_bottom2));
            corners[3]=Point((pfield[0].x+frac_top),(m_top*frac_top+pfield[0].y+frac_top2));
            corners[0]=Point(pfield[0].x,pfield[0].y);
                //cout<<"corner:"<<corners[0]<<corners[1]<<corners[2]<<corners[3]<<endl;
            mops_strips.push_back(corners);
        }
            //zwischenbereich
        if (i<div-1){
            Point* corners;
            corners=new Point[4];
            corners[1]=Point((pfield[1].x+i*frac_bottom),(m_bottom*i*frac_bottom+pfield[1].y+i*frac_bottom2));
            corners[2]=Point((pfield[1].x+(i+1)*frac_bottom),(m_bottom*(i+1)*frac_bottom+pfield[1].y+(i+1)*frac_bottom2));
            corners[3]=Point((pfield[0].x+(i+1)*frac_top),(m_top*(i+1)*frac_top+pfield[0].y+(i+1)*frac_top2));
            corners[0]=Point((pfield[0].x+i*frac_top),(m_top*i*frac_top+pfield[0].y+i*frac_top2));
            mops_strips.push_back(corners);
                //cout<<"corner:"<<corners[0]<<corners[1]<<corners[2]<<corners[3]<<endl;

        }
            //rechte Ecke definieren
        if (i==div-1){
            Point* corners;
            corners=new Point[4];
            corners[1]=Point((pfield[1].x+(div-1)*frac_bottom),(m_bottom*(div-1)*frac_bottom+pfield[1].y+(div-1)*frac_bottom2));
            corners[2]= Point(pfield[2].x,pfield[2].y);
            corners[3]=Point(pfield[3].x,pfield[3].y);
            corners[0]=Point((pfield[0].x+(div-1)*frac_top),(m_top*(div-1)*frac_top+pfield[0].y+(div-1)*frac_top2));
            //cout<<"corner:"<<corners[0]<<corners[1]<<corners[2]<<corners[3]<<endl;
            mops_strips.push_back(corners);
        }
    }
    return mops_strips;
}
	//**************************************************************************************
    // turns the parkfield clockwise
vector<Point> turnPoints(vector<Point> normal){
    
    vector<Point> turned;
    turned.clear();
    for (int i = 0; i < 4; i++)turned.push_back(Point(0, 0));
    turned[0].x=normal[3].x;
    turned[0].y=normal[3].y;
    turned[1].x=normal[0].x;
    turned[1].y=normal[0].y;
    turned[2].x=normal[1].x;
    turned[2].y=normal[1].y;
    turned[3].x=normal[2].x;
    turned[3].y=normal[2].y;
    return turned;
}
    //**************************************************************************************
    // get histogram
vector <int> get_histo(Mat frame,float& mean, float& stdev){
    float sum = 0.0;
    mean=0.0;
    stdev = 0.0;
    Mat frame_his;
    cvtColor(frame, frame_his, cv::COLOR_BGR2GRAY);
    equalizeHist( frame_his, frame_his );

    vector <int> cnts;
    int gray;
    for (int i=0;i<255;i++){cnts.push_back(0);}
    	for(int x = 0; x < frame_his.cols; x++) {
        	for(int y = 0; y < frame_his.rows; y++) {
            	gray = (int)frame_his.at<uchar>(y,x);
                cnts[gray]++;
            }
        }
    float sum_h,mean_h,stdev_h,sum_u,mean_u,stdev_u;
    for(int i = 0; i < cnts.size(); ++i) {
        
        sum += cnts[i];
            //cout<<cnts[i]<<endl;
        if (i==127){
            sum_h=sum;
        }
        if(i==cnts.size()){
            sum_u=sum-sum_h;
        }
    }
    mean = sum/cnts.size();
    mean_h=sum_h/128;
    mean_u=sum_u/128;
    
    for(int i = 0; i < cnts.size(); ++i) {
        stdev += pow(float(cnts[i] - mean), 2);
    }
    for(int i = 0; i < 127; ++i) {
        stdev_h += pow(float(cnts[i] - mean_h), 2);
    }
    for(int i = 128; i < 255; ++i) {
        stdev_u += pow(float(cnts[i] - mean_u), 2);
    }
    stdev=sqrt(stdev / (cnts.size()-1));
    stdev_h=sqrt(stdev_h / 127);
    stdev_u=sqrt(stdev_u / 127);
        //cout<<"s"<<stdev<<"h"<<stdev_h<<"u"<<stdev_u<<endl;
        //imshow("nachher",frame_his);
    return cnts;
}

    //**************************************************************************************
    // extract Spectrum
Mat extr_Spec(Mat frame,int low,int high){
    Mat frame1,frame2,frame_bw;
        //cvtColor(frame, frame_bw, cv::COLOR_BGR2GRAY);
    frame.copyTo(frame_bw);
    threshold(frame_bw, frame1, low, 255,0 );
    threshold(frame_bw, frame2, high, 255,1 );
    bitwise_and(frame1, frame2, frame1);
	namedWindow( "dada", WINDOW_AUTOSIZE);
//    imshow("dada",frame1);
//    waitKey(3999);
    return frame1;
}
    //**************************************************************************************
    // perspective Transformation

Mat four_point_transform(Mat frame,vector<Point>pts,vector<Point>& pfcorners_normal_warped,Mat& M){
        // Input Quadilateral or Image plane coordinates
    Point2f inputQuad[4];
        // Output Quadilateral or World plane coordinates
    Point2f outputQuad[4];
    Point2f outputQuad1[4];
    pfcorners_normal_warped.clear();
        //cout<<pts<<" eingang trans"<<endl;
    int width_top=sqrt(pow((pts[3].x-pts[0].x),2)+pow((pts[3].y-pts[0].y),2));
    int width_bottom=sqrt(pow((pts[2].x-pts[1].x),2)+pow((pts[2].y-pts[1].y),2));
    int maxwidth=width_top;
    int minwidth=width_bottom;
    if(width_bottom>width_top){maxwidth=width_bottom;minwidth=width_top;}
    int hight_left=sqrt(pow((pts[0].x-pts[1].x),2)+pow((pts[0].y-pts[1].y),2));
    int hight_right=sqrt(pow((pts[2].x-pts[3].x),2)+pow((pts[2].y-pts[3].y),2));
    int maxhight=hight_left;
    int minhight=hight_right;
    if(hight_right>hight_left){maxhight=hight_right;minhight=hight_left;}
    double width_ver=maxwidth/minwidth;
    double hight_ver=maxhight/minhight;
    pfcorners_normal_warped.push_back(Point(0,0));
    pfcorners_normal_warped.push_back(Point(0,maxhight-1));
    pfcorners_normal_warped.push_back(Point(maxwidth-1,maxhight-1));
    pfcorners_normal_warped.push_back(Point(maxwidth-1,0));
    inputQuad[0]=pts[0];
    inputQuad[1]=pts[3];
    inputQuad[2]=pts[2];
    inputQuad[3]=pts[1];
        //cout<<inputQuad[0]<<" "<<inputQuad[1]<<" "<<inputQuad[2]<<" "<<inputQuad[3]<<endl;
//    outputQuad[0]=Point(pfcorners_normal_warped[0].x+50,pfcorners_normal_warped[0].y+50);
//    outputQuad[1]=Point(pfcorners_normal_warped[3].x-50,pfcorners_normal_warped[3].y+50);
//    outputQuad[2]=Point(pfcorners_normal_warped[2].x-50,pfcorners_normal_warped[2].y-50);
//    outputQuad[3]=Point(pfcorners_normal_warped[1].x+50,pfcorners_normal_warped[1].y-50);
    outputQuad[0]=Point(pfcorners_normal_warped[0].x,pfcorners_normal_warped[0].y);
    outputQuad[1]=Point(pfcorners_normal_warped[3].x,pfcorners_normal_warped[3].y);
    outputQuad[2]=Point(pfcorners_normal_warped[2].x,pfcorners_normal_warped[2].y);
    outputQuad[3]=Point(pfcorners_normal_warped[1].x,pfcorners_normal_warped[1].y);
        //cout<<outputQuad[0]<<" "<<outputQuad[1]<<" "<<outputQuad[2]<<" "<<outputQuad[3]<<endl;
        // Set the lambda matrix the same type and size as input
        //Mat M2( 2, 4, CV_32FC1 );
    M=getPerspectiveTransform(inputQuad,outputQuad);
        //M2=getPerspectiveTransform(inputQuad,outputQuad1);
    Mat warped;
    
    warpPerspective(frame,warped,M,Size(maxwidth-1,maxhight-1));
        Mat frame_warpe;
        warpPerspective(frame,frame_warpe,M,Size(frame.cols*width_ver,frame.rows*hight_ver));
    float mean,stdev;
    vector<int> warped_histo=get_histo(warped,mean,stdev);

        //Mat warped_extr=extr_Spec(frame_his1,2,220);

        //cout<<"mean"<<mean<<" stdev:"<<stdev<<endl;
        //waitKey(89090000);
    return warped;
}
    //**************************************************************************************
    // calc mopsstrips, each PF is divided into strips, here we calc their mopsvalues

vector<double> calcmopsstrips(Mat overlay,Mat frame, vector<Point*> mops_strips,double reldev_pf,int mopstreshhold){
    vector<double> mops_strips_result;
    int nofps=0;
    int freestrips;
    bool statuschange=0;
    Mat bwframe;
    cv::cvtColor(frame, bwframe, cv::COLOR_BGR2GRAY);
    for(int i=0;i<mops_strips.size();i++){
        Mat mask = Mat(bwframe.size(), CV_8UC1, Scalar(0,0,0));
        fillConvexPoly(mask,mops_strips[i],4,Scalar(255,255,255));
        double reldev_strip=calreldev(frame,mask);
        double mops_strip = (double) (100*reldev_strip/reldev_pf);
//          cout<<reldev_strip<<" reldev_strip"<<endl;
//          cout<<mops_strip<<" mopsstrip"<<endl;
//        	cout<<reldev_pf<<" reldev_pf"<<endl;
            //coloring
        Scalar color( ((int)128*mops_strip/mopstreshhold),0,0);
        
            //cout<<128*mops_strip /mopstreshhold<<" color "<<mopstreshhold<<"mops: "<<mops_strip<<endl;
        fillConvexPoly(overlay,mops_strips[i],4,color);
        mops_strips_result.push_back(mops_strip);
        
//            imshow("howitlooks",overlay);
//            waitKey(5000);
    }
    return mops_strips_result;

}
    //**************************************************************************************
    // calciffree looks at the results of an mops_strip array and calc if the spot is free

int calciffree_1b1(vector<double> mops_strips_result,float striplength,int mopstreshhold,int carlength){
    double mean=0;
    int free1=0;
    int free=0;
    
    double freeofblue=0;
    for(int i=0;i<mops_strips_result.size();i++){
        double color;
        color=(double) mops_strips_result[i]*128/mopstreshhold;
        if(color>128){
            freeofblue+=0;
        }
        else{
            freeofblue+=1;
        }
        mean+=mops_strips_result[i];
	}
    freeofblue=(double)freeofblue/mops_strips_result.size();
    if (freeofblue>0.5){free1=1;}else{free1=0;}
    
    mean=mean/mops_strips_result.size();
    
    if (((mean/mopstreshhold)*128)<128){
        free=1;
    }
    else{free=0;}
    
    return free;
}

    //**************************************************************************************
    // calcfreespots looks at the results of an mops_strip array and calc how many spots are available

int calcfreespots(vector<double> mops_strips_result,float striplength,int mopstreshhold,int carlength){
    int freeiar=0;
    int blockiar=0;
    int uebertragfreeiar=0;
    int uebertragblockiar=0;
    int changer=2;
    int sumnfps=0;
    int ncarsf=0;
    striplength=abs(striplength);
    for(int i=0;i<mops_strips_result.size();i++){
        
        if(mops_strips_result[i]<mopstreshhold){
            if (blockiar>=3){	//3 noch anpassen XYXY
                freeiar=1;
                uebertragblockiar=blockiar;
                blockiar=0;
                changer=1;	//ab jetzt frei
            }
            else{freeiar++;changer=0;}
        }else{
            if (freeiar>=3){
                blockiar=1;
                uebertragfreeiar=freeiar;
                freeiar=0;
                changer=2;	//ab jetzt blockiert
            }
            else{blockiar++;changer=0;}
        }

//        if (i==div-1&&freeiar>blockiar){changer=3;  //letzte Parkzeile testen ob frei
//            ncarsf= (int) (freeiar*abs(fraca+fracb)/2)/carlength;
//            
//            text="fh "+to_string(ncarsf);
//            sumf=sumf+ncarsf;}
        if (changer==1){                               //wenn veränderung, gucken ob länge eines fahrzeuges entspricht
            ncarsf= (int) (uebertragblockiar/(carlength/striplength));
            if(ncarsf==0){changer=0;}
        }
        if (changer==2){
            ncarsf= (int) (uebertragfreeiar/(carlength/striplength));
            if(ncarsf==0){changer=0;}
                //cout<<"free "<<ncarsf<<"carlength "<<carlength/striplength<<";striplength "<<striplength<<"no. strips: "<<mops_strips_result.size()<<endl;
            sumnfps=sumnfps+ncarsf;
        }
        if (i==mops_strips_result.size()-1&&freeiar>blockiar){changer=3;  //letzte Parkzeile testen ob frei
            ncarsf= (int) (freeiar/(carlength/striplength));
                //cout<<"free "<<ncarsf<<"carlength "<<carlength/striplength<<";striplength "<<striplength<<"no. strips: "<<mops_strips_result.size()<<endl;
            sumnfps=sumnfps+ncarsf;}
        
        uebertragfreeiar=0;
        uebertragblockiar=0;
        
    }
    
        return sumnfps;
}
