using Emgu.CV;
using System.Drawing;

namespace EmguExtensions.Tests;

public static class MatFactory
{
    internal static Mat HalfBlackWhiteMat
    {
        get
        {
            var mat = EmguExtensions.InitMat(new Size(100, 100));
            CvInvoke.Rectangle(mat, new Rectangle(50, 50, 50, 50), EmguExtensions.WhiteColor, -1);

            return mat;
        }
    }
}