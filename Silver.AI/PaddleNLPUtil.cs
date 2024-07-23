using Sdcb.PaddleNLP.Lac;

namespace Silver.AI
{
    /// <summary>
    /// 分词
    /// </summary>
    public class PaddleNLPUtil:IDisposable
    {
        private ChineseSegmenter chineseSegmenter;
        public PaddleNLPUtil()
        {
            this.chineseSegmenter = new ChineseSegmenter();
        }

        public PaddleNLPUtil(Dictionary<string, WordTag?> defaultKey)
        {
            this.chineseSegmenter = new ChineseSegmenter(new LacOptions(defaultKey));
        }

        /// <summary>
        /// 分词
        /// </summary>
        /// <param name="keyWord"></param>
        /// <returns></returns>
        public List<string> Participle(string keyWord)
        {
            return chineseSegmenter.Segment(keyWord).ToList();
        }

        /// <summary>
        /// 分词
        /// </summary>
        /// <param name="keyWord"></param>
        /// <returns></returns>
        public List<WordAndTag> ParticipleTag(string keyWord)
        {
           return chineseSegmenter.Tagging(keyWord).ToList();
        }

        public void Dispose()
        {
            if (this.chineseSegmenter != null)
            {
                this.chineseSegmenter.Dispose();
            }
        }
    }
}
