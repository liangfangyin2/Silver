using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace Silver.AI
{
    public class OnnxUtil : IDisposable
    {
        private InferenceSession inferenceSession;
        private List<string> inputMetaKeys= new List<string>();
        public OnnxUtil()
        {
            this.inferenceSession = new InferenceSession(System.IO.Directory.GetCurrentDirectory() + "/model.onnx");
            this.inputMetaKeys = this.inferenceSession.InputMetadata.Keys.ToList();
        }

        public OnnxUtil(string path)
        {
            this.inferenceSession = new InferenceSession(path);
            this.inputMetaKeys = this.inferenceSession.InputMetadata.Keys.ToList();
        }

        /// <summary>
        /// 识别图片
        /// </summary>
        /// <param name="imagePath">图片</param>
        /// <returns></returns>
        public float[] RecognitionImage(string imagePath)
        {
            using (Mat imgMat = new Mat(imagePath))
            {
                imgMat.Resize(new Size(224,224));
                Mat resizemat = new Mat();
                Cv2.CvtColor(imgMat, resizemat, ColorConversionCodes.BGR2RGB);
                var inputTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
                for (int y = 0; y < resizemat.Height; y++)
                {
                    for (int x = 0; x < resizemat.Width; x++)
                    {
                        inputTensor[0, 0, y, x] = (resizemat.At<Vec3b>(y, x)[0] / 255f - 0.5f) / 0.5f;
                        inputTensor[0, 1, y, x] = (resizemat.At<Vec3b>(y, x)[1] / 255f - 0.5f) / 0.5f;
                        inputTensor[0, 2, y, x] = (resizemat.At<Vec3b>(y, x)[2] / 255f - 0.5f) / 0.5f;
                    }
                }
                // 准备输入
                var inputs = new List<NamedOnnxValue>{
                   NamedOnnxValue.CreateFromTensor(inputMetaKeys.FirstOrDefault(), inputTensor)
                };
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = inferenceSession.Run(inputs))
                {
                    var resultsArray = results.ToArray();
                    var dimensions = resultsArray[0].AsTensor<float>().Dimensions;
                    return resultsArray[0].AsEnumerable<float>().ToArray();
                }
            }
        }

        /// <summary>
        /// 识别图片
        /// </summary>
        /// <param name="imagePath">图片</param>
        /// <param name="inputTensor">输入参数</param>
        /// <param name="width">图片宽度</param>
        /// <param name="height">图片高度</param>
        /// <returns></returns>
        public float[] RecognitionImage(string imagePath, DenseTensor<float> inputTensor, int width=224,int height=224)
        {
            using (Mat imgMat = new Mat(imagePath))
            {
                imgMat.Resize(new Size(width, height));
                Mat resizemat = new Mat();
                Cv2.CvtColor(imgMat, resizemat, ColorConversionCodes.BGR2RGB); 
                var inputs = new List<NamedOnnxValue>{
                   NamedOnnxValue.CreateFromTensor(inputMetaKeys.FirstOrDefault(), inputTensor)
                };
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = inferenceSession.Run(inputs))
                {
                    var resultsArray = results.ToArray();
                    var dimensions = resultsArray[0].AsTensor<float>().Dimensions;
                    return resultsArray[0].AsEnumerable<float>().ToArray();
                }
            }
        }

        /// <summary>
        /// 识别图片
        /// </summary>
        /// <param name="imageBytes">图片Byte</param>
        /// <param name="inputTensor">输入参数</param>
        /// <param name="width">图片宽度</param>
        /// <param name="height">图片高度</param>
        /// <returns></returns>
        public float[] RecognitionImage(byte[] imageBytes, DenseTensor<float> inputTensor, int width = 224, int height = 224)
        {
            using (Mat imgMat = Mat.ImDecode(imageBytes))
            {
                imgMat.Resize(new Size(width, height));
                Mat resizemat = new Mat();
                Cv2.CvtColor(imgMat, resizemat, ColorConversionCodes.BGR2RGB);
                var inputs = new List<NamedOnnxValue>{
                   NamedOnnxValue.CreateFromTensor(inputMetaKeys.FirstOrDefault(), inputTensor)
                };
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = inferenceSession.Run(inputs))
                {
                    var resultsArray = results.ToArray();
                    var dimensions = resultsArray[0].AsTensor<float>().Dimensions; 
                    return resultsArray[0].AsEnumerable<float>().ToArray();
                }
            }
        }

        /// <summary>
        /// 释放
        /// </summary>
        public void Dispose()
        {
            if (this.inferenceSession != null)
            {
                this.inferenceSession.Dispose();
            }
        }
    }
}
