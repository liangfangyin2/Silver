using OpenCvSharp;
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR.Models.Local;

namespace Silver.AI
{
    /// <summary>
    /// Ocr文字识别
    /// </summary>
    public class PaddleOCRUtil : IDisposable
    {
        private FullOcrModel model;
        private PaddleOcrAll paddleOcrAll;
        public PaddleOCRUtil()
        {
            this.model = LocalFullModels.ChineseV3;
            this.paddleOcrAll = new PaddleOcrAll(this.model, PaddleDevice.Mkldnn())
            {
                AllowRotateDetection = true, /* 允许识别有角度的文字 */
                Enable180Classification = false, /* 允许识别旋转角度大于90度的文字 */
            };
        }

        public PaddleOCRUtil(FullOcrModel _model)
        {
            this.model = _model;
            this.paddleOcrAll = new PaddleOcrAll(this.model, PaddleDevice.Mkldnn())
            {
                AllowRotateDetection = true, /* 允许识别有角度的文字 */
                Enable180Classification = false, /* 允许识别旋转角度大于90度的文字 */
            };
        }

        /// <summary>
        /// Ocr识别
        /// </summary>
        /// <param name="imagePath"></param>
        /// <returns></returns>
        public PaddleOcrResult Recognition(string imagePath)
        {
            var imageByte = File.ReadAllBytes(imagePath);
            return Recognition(imageByte);
        }

        /// <summary>
        /// Ocr识别
        /// </summary>
        /// <param name="imageByte"></param>
        /// <returns></returns>
        public PaddleOcrResult Recognition(byte[] imageByte)
        {
            using (Mat src = Cv2.ImDecode(imageByte, ImreadModes.Color))
            {
                return paddleOcrAll.Run(src);
            }
        }

        public void Dispose()
        {
            if (paddleOcrAll != null)
            {
                paddleOcrAll.Dispose();
            }
        }
    }
}
