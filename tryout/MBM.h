#include <opencv2/opencv.hpp>

class MBM {
private:
	static inline int absdiff(const cv::Vec3b &o1, const cv::Vec3b &o2) {
		int l1 = std::abs(o1[0] - o2[0]);
		int l2 = std::abs(o1[1] - o2[1]);
		int l3 = std::abs(o1[2] - o2[2]);
		return l1 + l2 + l3;
	}

public:
	static cv::Mat compute(cv::Mat l, cv::Mat r, int mindisp, int ndisp) {
		cv::Mat disparity(r.rows, r.cols, CV_16S);
		disparity = 0;
		cv::Mat corrs(r.rows, r.cols, CV_16U);
		corrs = 1000000;
		cv::Mat corrs2(r.rows, r.cols, CV_16U);
		corrs2 = 1000000;
		cv::Mat eimg(r.rows, r.cols, CV_32S);

		for (int d = mindisp; d < mindisp + ndisp; d++) {

			int xfrom = std::max(0, - d);
			int xto = std::min(r.cols, r.cols - d);
			//eimg = 0;

			for (int y = 0; y < r.rows; y++) {
				const cv::Vec3b* rptr = r.ptr<cv::Vec3b>(y);
				const cv::Vec3b* lptr = l.ptr<cv::Vec3b>(y);
				int* eptr = eimg.ptr<int>(y);

				for (int x = xfrom; x < xto; x++) {
					cv::Vec3b rv = rptr[x];
					cv::Vec3b lv = lptr[x + d];
					eptr[x] = absdiff(rv, lv);
				}
			}

			/*
			cv::Mat g;
			cv::normalize(eimg, g, 0, 255, cv::NORM_MINMAX, CV_8U);
			cv::imshow("h", g);
			std::cout << d << std::endl;
			cv::waitKey(0);
			*/

			int* leptr = nullptr;
			for (int y = 0; y < r.rows; y++) {
				int* eptr = eimg.ptr<int>(y);

				for (int x = xfrom; x < xto; x++) {
					int nval = eptr[x];
					if (x > xfrom) {
						nval += eptr[x - 1];
					}
					if (y > 0) {
						nval += leptr[x];
					}
					if (x > xfrom && y > 0) {
						nval -= leptr[x - 1];
					}
					eptr[x] = nval;
				}

				leptr = eptr;
			}

			int bsX[] = {5, 10, 5, 60, 2};
			int bsY[] = { 5, 5, 10, 2, 60};
			int nb = 3;
			for (int y = 0; y < r.rows; y++) {

				for (int x = xfrom; x < xto; x++) {
					int glob_ev = 0;
					for (int i = 0; i < nb; i++) {
						int ev = 0;
						int cbX = bsX[i];
						int cbY = bsY[i];
						int yUp = y - cbY;
						int yDown = y + cbY;
						int xLeft = x - cbX;
						int xRight = x + cbX;

						if (yUp >= 0 && xLeft >= xfrom) {
							ev += eimg.at<int>(yUp, xLeft);
						}
						if (yUp >= 0 && xRight < xto) {
							ev -= eimg.at<int>(yUp, xRight);
						}
						if (yDown < r.rows && xLeft >= xfrom) {
							ev -= eimg.at<int>(yDown, xLeft);
						}
						if (yDown < r.rows && xRight < xto) {
							ev += eimg.at<int>(yDown, xRight);
						}

						ev /= (cbX * cbY);
						glob_ev += ev;
					}
					if (corrs.at<ushort>(y, x) > glob_ev) {
						disparity.at<short>(y, x) = d;
						corrs2.at<ushort>(y, x) = corrs.at<ushort>(y, x);
						corrs.at<ushort>(y, x) = glob_ev;
					}
				}
			}
		}

		for (int y = 0; y < disparity.rows; y++) {
			for (int x = 0; x < disparity.cols; x++) {
				if ((float)corrs.at<ushort>(y, x) * 1.3 > (float)corrs2.at<ushort>(y, x)) {
					disparity.at<short>(y, x) = 0;
				}
			}
		}

		return disparity;
	}
};