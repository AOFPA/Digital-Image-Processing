#include <opencv2/opencv.hpp>
// Header for write files
#include <ios>
#include <fstream>
#include <iostream>

#include <omp.h>

using namespace cv;
using namespace std;

//Global variable
int IN_number=0;
int OUT_number=0;
long int FrameID = 0;

/**********************************************************************
 *  ConvertTo String function
 *
 **********************************************************************/
string intToStr(int num)
{
	ostringstream convert;   // stream used for the conversion
	convert << num;      // insert the textual representation of 'Number' in the characters in the stream
	return convert.str(); // set 'Result' to the contents of the stream
}
string floatToStr(float num)
{
	ostringstream convert;   // stream used for the conversion
	convert << num;      // insert the textual representation of 'Number' in the characters in the stream
	return convert.str(); // set 'Result' to the contents of the stream
}
string charToStr(char num)
{
	ostringstream convert;   // stream used for the conversion
	convert << num;      // insert the textual representation of 'Number' in the characters in the stream
	return convert.str(); // set 'Result' to the contents of the stream
}

class blob
{
public:
	int id, x, y, yBefore1, yBefore2, width, height, lifes, fullLifeValue;
	Scalar color;
	RNG rng15;
	char Direction;
	bool countedFlag, aliveFlag;
	blob(int idIndex, int initLife, int xCurrent, int yCurrent, int widthCurrent, int heightCurrent)
	{
		id = idIndex;
		lifes = initLife;
		fullLifeValue = initLife;
		yBefore2 = 0;
		yBefore1 = 0;
		y = yCurrent;
		x = xCurrent;
		width = widthCurrent;
		height = heightCurrent;
		countedFlag = false;
		aliveFlag = true;
		Direction = 'U';
		rng15(12345);
		color = Scalar(rng15.uniform(0, 255), rng15.uniform(0, 255), rng15.uniform(0, 255));
	}
	void updateBlob(bool status, int xCurrent = 0, int yCurrent = 0, int widthCurrent = 0, int heightCurrent = 0)
	{
		if (status)
		{
			lifes = fullLifeValue;
			yBefore2 = yBefore1;
			yBefore1 = yCurrent;
			y = yCurrent;
			x = xCurrent;
			width = widthCurrent;
			height = heightCurrent;
			if (yBefore2 != 0 && yBefore1 != 0)
			{
				if (((yBefore2 - yBefore1) + (yBefore1 - y)) > 0) { Direction = 'U'; }
				else { Direction = 'D'; }
			}
		}
		else
		{
			lifes--;
			if (lifes == 0) { aliveFlag = false; }
		}
	}
	float calErrorWithContour(int xContour, int yContour, int widthContour, int heightContour, float positionWeight = 0.5, float scaleWeight = 0.5)
	{

		return ((float)sqrt(pow((xContour - x), 2) + pow((yContour - y), 2)) * positionWeight) + ((float)sqrt(pow((widthContour - width), 2) + pow((heightContour - height), 2)) * scaleWeight);

	}
	void setCountedFlag() { countedFlag = true; }
	bool getCountedFlag() { return countedFlag; }
	bool alive() { return aliveFlag; }
	char getDirection() { return Direction; }

};

void adjustThreshold(int, void*);

void adjustThreshold(int, void*)
{

}


void counter(Mat& objectImage, Mat& originalImage, vector<blob>& trackingBlob, size_t& counterIn, size_t& counterOut, int widthOfCountingArea = 200)
{
	// Make area for counting
	size_t yMin = saturate_cast<size_t>(objectImage.rows / 3) - (widthOfCountingArea / 2);
	size_t yMax = saturate_cast<size_t>(objectImage.rows / 3) + (widthOfCountingArea / 2);
	size_t xMin = 0;
	size_t xMax = objectImage.cols - 1;
	line(objectImage, Point(xMin, yMin), Point(xMax, yMin), Scalar(255, 255, 255));
	line(objectImage, Point(xMin, yMax), Point(xMax, yMax), Scalar(255, 255, 255));
	// Access to Check never count trackingBlob
	for (int i = 0;i < trackingBlob.size();i++)
	{
		if (!trackingBlob[i].getCountedFlag())
		{
			if (trackingBlob[i].y >= yMin && trackingBlob[i].y <= yMax)
			{
				Mat tempWrite = originalImage.clone();
				//for write image
				Rect adjustRect(trackingBlob[i].x, trackingBlob[i].y, trackingBlob[i].width, trackingBlob[i].height);
				adjustRect.x += 0;
				adjustRect.y += 0;
				rectangle(tempWrite, adjustRect, Scalar(0, 40, 255), 3);
				time_t  timev;
				time(&timev);
				string datetime = ctime(&timev);
				putText(tempWrite, datetime, Point(180, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(120, 255, 0), 2);

				if (trackingBlob[i].Direction == 'U')
				{
					//counterOut++;
					OUT_number++;
					//Write Image
					putText(tempWrite, "[OUT" + intToStr(OUT_number) + "] At", Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(120, 255, 0), 2);
					imshow("OUT", tempWrite);

				}
				else
				{
					//counterIn++;
					IN_number++;
					//Write Image
					putText(tempWrite, "[IN" + intToStr(IN_number) + "] At", Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(120, 255, 0), 2);
					imshow("IN", tempWrite);
					

				}
				trackingBlob[i].setCountedFlag();
			}
		}
	}
	// Output number of counting
	ostringstream strConvertIn, strConvertOut;
	strConvertIn << "IN : " << IN_number;
	strConvertOut << "OUT : " << OUT_number;
	putText(objectImage, strConvertIn.str(), Point(10, objectImage.rows - 200), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
	putText(objectImage, strConvertOut.str(), Point(objectImage.cols - 150, objectImage.rows - 200), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);


}

void tracker(Mat& objectImage, vector<blob>& trackingBlob, vector<Rect> boundRectangle, int& idxLastCurrent, float positionWeightValue = 0.5, float scaleWeightValue = 0.5, int lifeCycle = 3, float errorThreshold = 60)
{
	bool newObjectFlag = true;
	size_t minErrorObjectIndex = 0;
	size_t minErrorObjectValue = 0;
	size_t ErrorObjectValue = 0;
	vector <bool> updatedList(trackingBlob.size());
	for (int a = 0;a < updatedList.size();a++) { updatedList[a] = false; }// Initial value of updated trackingBlob
	for (int i = 0;i < boundRectangle.size();i++) // Acess in the each of contour
	{
		//cout <<endl<<"-----------------------------------------------------------------------"<< endl <<"Process with Contour->" << i;
		newObjectFlag = true;//Refresh Flag
		// Calculate Score;
		for (int j = 0;j < trackingBlob.size();j++)
		{

			ErrorObjectValue = trackingBlob[j].calErrorWithContour(boundRectangle[i].x, boundRectangle[i].y, boundRectangle[i].width, boundRectangle[i].height, positionWeightValue, scaleWeightValue);
			//cout << endl <<"Contour-> " << i <<" compared error with trackingBlob->" << j <<" is "<<ErrorObjectValue;
			if (j == 0)//Initial first value of min
			{
				minErrorObjectValue = ErrorObjectValue;
				if (ErrorObjectValue < errorThreshold)
				{
					minErrorObjectIndex = 0;
					newObjectFlag = false;
				}
			}
			else if (ErrorObjectValue < errorThreshold && ErrorObjectValue <= minErrorObjectValue)
			{

				minErrorObjectValue = ErrorObjectValue;
				minErrorObjectIndex = j;
				newObjectFlag = false;
			}
		}
		// Update Or new obj
		if (newObjectFlag)
		{

			blob newBlob(++idxLastCurrent, lifeCycle, boundRectangle[i].x, boundRectangle[i].y, boundRectangle[i].width, boundRectangle[i].height);
			trackingBlob.push_back(newBlob);
			updatedList.push_back(true);
			//cout <<endl<< "Added new trackingBlob->"<<trackingBlob.size()-1;
		}
		else
		{
			//cout <<endl<< "Updated trackingBlob->"<<minErrorObjectIndex;
			trackingBlob[minErrorObjectIndex].updateBlob(true, boundRectangle[i].x, boundRectangle[i].y, boundRectangle[i].width, boundRectangle[i].height);
			updatedList[minErrorObjectIndex] = true;
		}

	}


	// Set value not updated trackingBlob
	for (int a = 0;a < updatedList.size();a++)
	{
		if (updatedList[a] == false)trackingBlob[a].updateBlob(false);
	}

	// Remove dead 
	size_t sizeBlob = trackingBlob.size();
	for (int b = 0;b < sizeBlob;b++)
	{
		if (!trackingBlob[b].alive())
		{
			//cout<<endl<<"Removed trackingBlob->"<<b;
			trackingBlob.erase(trackingBlob.begin() + b);
			b--;
			sizeBlob--;

		}
	}

	//Output of tracking
	vector<ostringstream> strConvert(trackingBlob.size());
	for (int j = 0;j < trackingBlob.size();j++)
	{
		//Output ID
		strConvert[j] << trackingBlob[j].id;
		putText(objectImage, strConvert[j].str(), Point(trackingBlob[j].x + trackingBlob[j].width, trackingBlob[j].y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
		//Output Direction
		if (trackingBlob[j].getDirection() == 'U')
		{
			line(objectImage, Point(trackingBlob[j].x + (trackingBlob[j].width / 2), trackingBlob[j].y + (trackingBlob[j].height / 2)), Point(trackingBlob[j].x + (trackingBlob[j].width / 2), trackingBlob[j].y), Scalar(255, 255, 255), 2);
		}
		else
		{
			line(objectImage, Point(trackingBlob[j].x + (trackingBlob[j].width / 2), trackingBlob[j].y + (trackingBlob[j].height / 2)), Point(trackingBlob[j].x + (trackingBlob[j].width / 2), trackingBlob[j].y + trackingBlob[j].height), Scalar(255, 255, 255), 2);
		}
	}
}




int getCtrlPoint(Mat srcBinary, Mat& desContourImg, vector<vector<Point>>& contours, vector<Rect>& boundRectangle, vector<Vec4i>& hierarchy, size_t epsilon, size_t terminate_size, int retrieval)
{
	RNG rng15(12345);


	if (srcBinary.channels() != 1)
	{
		return 0;
	}

	/// Find contours  
	findContours(srcBinary, contours, hierarchy, retrieval, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], epsilon, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle(contours_poly[i], center[i], radius[i]);
	}

	size_t contour_size = contours.size();
	for (int i = 0; i < contours.size(); i++)
	{
		//cout<<endl<<boundRect[i].width*boundRect[i].height<<endl;
		if (boundRect[i].width * boundRect[i].height <= terminate_size)
		{

			boundRect.erase(boundRect.begin() + i);
			contours.erase(contours.begin() + i);
			contours_poly.erase(contours_poly.begin() + i);
			contour_size--;
			i--;
		}

	}


	/// Draw polygonal contour + bonding rects + circles
	desContourImg = Mat::zeros(srcBinary.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng15.uniform(0, 255), rng15.uniform(0, 255), rng15.uniform(0, 255));

		drawContours(desContourImg, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(desContourImg, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		//circle( desContourImg, center[i], (int)radius[i], color, 2, 8, 0 );
		circle(desContourImg, Point(boundRect[i].x + (boundRect[i].width / 2), boundRect[i].y + (boundRect[i].height / 2)), 2, Scalar(0, 0, 255), 2, 8, 0);
	}


	boundRectangle.swap(boundRect);
	return 1; //Normal Exit
}

int filterLaplacian(Mat src, Mat& des)
{
	Mat src_gray, dst;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;


	int c;

	if (!src.data) { return -1; }


	/// Convert it to gray
	if (src.channels() != 1)cvtColor(src, src_gray, COLOR_BGR2GRAY);
	else src_gray = src.clone();

	/// Apply Laplace function
	Mat abs_dst;

	Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(dst, des);


	return 1; //Normal Exit
}




int main(int argc, char* argv[])
{
	Mat originalImg;

	// Open Video input
	VideoCapture inputColorVideo;
	if (argc >= 2)
	{
		inputColorVideo.open(argv[1]);
	}
	else
	{
		cout << "\nPlease input video by argv[1] passing.";
		return 1;
	}

	if (!inputColorVideo.isOpened())
	{
		cout << "Could not open the input video." << endl;
		return -1;
	}


	// Declare Variable
	int numLearningBG = 20;
	int trackingThresholdValue = 100;
	Mat grayImg, foregroundImg, contourImg, laplacianImg, colorImg, ori;
	vector<blob> trackingObjectList;
	vector<Rect> boundRectangleList;
	int indexOfNewObject = 0;
	size_t enterCounter = 0;
	size_t exitCounter = 0;

	//Background Substraction
	Ptr<BackgroundSubtractor> BackSub;
	BackSub = createBackgroundSubtractorMOG2(500,50,false);


	// Allocation Windows
	namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display
	namedWindow( "Display Gray Scale", WINDOW_NORMAL );// Create a window for display.
	namedWindow( "Display Edge from Laplacian", WINDOW_NORMAL );// Create a window for display.
	namedWindow( "Display Contour", WINDOW_NORMAL );// Create a window for display.
	namedWindow("Display ForeGround", WINDOW_NORMAL);// Create a window for display.
	namedWindow("Display Counting", WINDOW_NORMAL);// Create a window for display.
	namedWindow("IN", WINDOW_NORMAL);
	namedWindow("OUT", WINDOW_NORMAL);
	// Create TrackBar for adjust Binarization threshold & Tracking Error Theshold 
	//createTrackbar( "Binarization Threshold",	"Display Binary", &thresholdValue, 255, adjustThreshold );
	createTrackbar( "Tracking Error Threshold",	"Display Counting", &trackingThresholdValue, 200, adjustThreshold );

	char key;

	// Loop Access frame
	while (true) //Show the image captured in the window and repeat
	{
		key = waitKey(10);
		if (key == 27) break; //if press ESC = > exit

		double pstart = omp_get_wtime();
		boundRectangleList.clear();		// Clear bounding rectangle box

		inputColorVideo >> colorImg;              // Read Color frame
		if (colorImg.empty())			// check if at end
		{
			
			inputColorVideo.open(argv[1]);//loop video
			if ( !inputColorVideo.isOpened())
			{
				cout << "Could not open the input video." << endl;
				return -1;
			}
			else
			{
				inputColorVideo >> colorImg;              // Read Color frame
			}
			//continue;
		}
		cvtColor(colorImg, grayImg, COLOR_BGR2GRAY); // Convert to Gray Scale
		originalImg = colorImg.clone();
		//GaussianBlur(grayImg, grayImg, Size(3, 3), 3); //Smoothing Image

		if (FrameID <= numLearningBG || key == 'i') BackSub->apply(grayImg, foregroundImg, 0.5); //learing BG 
		else BackSub->apply(grayImg, foregroundImg,0.05); //substract BG
		

		// Morphology operation
		Mat element = getStructuringElement(MORPH_CROSS, Size(7, 7));
		morphologyEx(foregroundImg, foregroundImg, MORPH_CLOSE, element);
		//morphologyEx(binImg, binImg, MORPH_DILATE, element);

		// Edge Detection(Laplacian)
		filterLaplacian(foregroundImg.clone(), laplacianImg);

		// Get Control Point of Input
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		getCtrlPoint(laplacianImg.clone(), contourImg, contours, boundRectangleList, hierarchy, 3, 4000, RETR_EXTERNAL);

		// Tracking Function
		tracker(colorImg, trackingObjectList, boundRectangleList, indexOfNewObject, 0.6, 0.4, 3, trackingThresholdValue);

		// Counting Function
		counter(colorImg, originalImg, trackingObjectList, enterCounter, exitCounter);

		// Display with color video
		for (int i = 0;i < boundRectangleList.size();i++)
		{
			Rect adjustRect = boundRectangleList[i];
			adjustRect.x += 8;
			adjustRect.y += 8;
			rectangle(colorImg, adjustRect, Scalar(0, 255, 20));
		}

		double pstop = omp_get_wtime();
		cout << "\nProcesses used time " << (pstop - pstart) * 1000 << " ms";

		if(FrameID <= numLearningBG || key == 'i') putText(colorImg,"Learning Background...", Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 40, 0), 2);

		// Display Image of the each step
		imshow( "Display Gray Scale", grayImg ); 
		imshow( "Display ForeGround", foregroundImg);
		imshow( "Display Edge from Laplacian",laplacianImg );
		imshow( "Display Contour",contourImg );
		imshow("Display Counting", colorImg);
		FrameID++;
	}


	return 0;
}