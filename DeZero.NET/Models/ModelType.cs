namespace DeZero.NET.Models
{
    public enum ModelType
    {
        /// <summary>
        /// 未設定
        /// </summary>
        Unset = 0,

        /// <summary>
        /// 回帰モデル
        /// </summary>
        Regression,

        /// <summary>
        /// 分類モデル
        /// </summary>
        Classification,

        /// <summary>
        /// 生成モデル
        /// </summary>
        Generative,

        /// <summary>
        /// クラスタリングモデル
        /// </summary>
        Clustering,

        /// <summary>
        /// 強化学習モデル
        /// </summary>
        ReinforcementLearning,

        /// <summary>
        /// 次元削減モデル
        /// </summary>
        DimensionalityReduction,

        /// <summary>
        /// アンサンブルモデル
        /// </summary>
        Ensemble,
    }
}
