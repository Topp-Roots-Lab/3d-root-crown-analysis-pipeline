#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <limits>
#include <iostream>

using namespace cv;
using namespace std;

int Yen(MatND data)
{
    /* Ported to C++ by Alexander Ivakov from Java implementation of ImageJ plugin Auto_Threshold

        Implements Yen  thresholding method
        1) Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
           for Automatic Multilevel Thresholding" IEEE Trans. on Image
           Processing, 4(3): 370-378
        2) Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165
           http://citeseer.ist.psu.edu/sezgin04survey.html

        M. Emre Celebi
        06.15.2007
        Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
     */
    int threshold;
    int ih, it;
    double crit;
    double max_crit;
    double norm_histo[256]; /* normalized histogram */
    double P1[256];         /* cumulative normalized histogram */
    double P1_sq[256];
    double P2_sq[256];

    int total = 0;
    for (ih = 0; ih < 256; ih++)
    {
        total += (int)data.at<float>(ih);
    }
    // cout << "Total: " << total << endl;

    for (ih = 0; ih < 256; ih++)
    {
        norm_histo[ih] = (double)data.at<float>(ih) / total;
    }

    P1[0] = norm_histo[0];
    for (ih = 1; ih < 256; ih++)
    {
        P1[ih] = P1[ih - 1] + norm_histo[ih];
    }
    P1_sq[0] = norm_histo[0] * norm_histo[0];
    for (ih = 1; ih < 256; ih++)
    {
        P1_sq[ih] = P1_sq[ih - 1] + norm_histo[ih] * norm_histo[ih];
    }
    P2_sq[255] = 0.0;
    for (ih = 254; ih >= 0; ih--)
    {
        P2_sq[ih] = P2_sq[ih + 1] + norm_histo[ih + 1] * norm_histo[ih + 1];
    }

    /* Find the threshold that maximizes the criterion */
    threshold = -1;
    max_crit = std::numeric_limits<double>::min();

    for (it = 0; it < 256; it++)
    {
        crit = -1.0 * ((P1_sq[it] * P2_sq[it]) > 0.0 ? log(P1_sq[it] * P2_sq[it]) : 0.0) + 2 * ((P1[it] * (1.0 - P1[it])) > 0.0 ? log(P1[it] * (1.0 - P1[it])) : 0.0);
        if (crit > max_crit)
        {
            max_crit = crit;
            threshold = it;
        }
    }
    return threshold;
}