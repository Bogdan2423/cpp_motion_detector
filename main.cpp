#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    string src = "/home/bogdan/CLionProjects/motion_detector1/lublin.mp4";
    int sens = 50;
    bool debug = true;
    int mask_coordinates[] = {100, 100, 1000, 1000};

    Mat frame, gray, frameDelta, thresh, firstFrame, mask, masked;
    vector<vector<Point> > contours;

    VideoCapture camera;
    camera.open(src);
    camera.read(firstFrame);

    mask = Mat::zeros(firstFrame.size(), firstFrame.type());
    rectangle(mask,
              Point(mask_coordinates[0], mask_coordinates[1]),
              Point(mask_coordinates[2], mask_coordinates[3]),
              Scalar(255, 255, 255), -1);

    bitwise_and(mask, firstFrame, firstFrame);
    resize(firstFrame, firstFrame, Size(800,600));
    cvtColor(firstFrame, firstFrame, COLOR_BGR2GRAY);
    GaussianBlur(firstFrame, firstFrame, Size(15, 15), 0);

    while(camera.read(frame)) {
        bitwise_and(mask, frame, masked);

        resize(frame, frame, Size(800,600));
        resize(masked, masked, Size(800,600));

        cvtColor(masked, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(15, 15), 0);

        absdiff(firstFrame, gray, frameDelta);
        threshold(frameDelta, thresh, sens, 255, THRESH_BINARY);
        dilate(thresh, thresh, Mat(), Point(-1,-1), 2);
        findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<Rect> boundRect(contours.size());
        int j = 0;
        for(auto & contour : contours) {
            if(contourArea(contour) >= 500) {
                boundRect[j] = boundingRect(contour);
                j++;
            }
        }

        Scalar color = Scalar(0, 255, 0);
        for (int i = 0; i<j; i++)
            rectangle(frame, boundRect[i].tl(), boundRect[i].br(), color, 2);

        imshow("Camera", frame);
        if (debug){
            imshow("Masked", masked);
            imshow("Grayscale", gray);
            imshow("Delta", frameDelta);
            imshow("Thresh", thresh);
        }

        if(waitKey(1) == 'q')
            break;
    }

    return 0;
}