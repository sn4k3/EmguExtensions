using Emgu.CV;
using System.Drawing;

namespace EmguExtensions.Tests;

public static class MatFactory
{
    internal static Mat HalfBlackWhiteMat
    {
        get
        {
            var mat = EmguCvExtensions.InitMat(new Size(100, 100));
            CvInvoke.Rectangle(mat, new Rectangle(50, 50, 50, 50), EmguCvExtensions.WhiteColor, -1);

            return mat;
        }
    }
}