#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    string src = "/home/bogdan/CLionProjects/motion_detector1/lublin.mp4";
    int sens = 50;
    bool debug = false;

    Mat frame, gray, frameDelta, thresh, firstFrame;
    vector<vector<Point> > contours;
    VideoCapture camera;
    camera.open(src);
    camera.read(firstFrame);
    resize(firstFrame, firstFrame, Size(800,600));

    cvtColor(firstFrame, firstFrame, COLOR_BGR2GRAY);
    GaussianBlur(firstFrame, firstFrame, Size(21, 21), 0);

    while(camera.read(frame)) {
        resize(frame, frame, Size(800,600));
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(21, 21), 0);
        absdiff(firstFrame, gray, frameDelta);
        threshold(frameDelta, thresh, sens, 255, THRESH_BINARY);
        dilate(thresh, thresh, Mat(), Point(-1,-1), 2);
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<Rect> boundRect(contours.size());
        int j = 0;
        for(int i = 0; i < contours.size(); i++) {
            if(contourArea(contours[i]) >= 500) {
                boundRect[j] = boundingRect(contours[i]);
                j++;
            }
        }

        Scalar color = Scalar(0, 255, 0);
        for (int i = 0; i<j; i++){
            rectangle(frame, boundRect[i].tl(), boundRect[i].br(), color, 2);
        }

        imshow("Camera", frame);
        if (debug){
            imshow("Grayscale", gray);
            imshow("Delta", frameDelta);
            imshow("Thresh", thresh);
        }

        if(waitKey(1) == 'q')
            break;
    }

    return 0;
}