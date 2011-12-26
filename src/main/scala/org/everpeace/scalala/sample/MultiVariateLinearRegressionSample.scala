package org.everpeace.scalala.sample

import scala.io.Source.fromFile
import scalala.tensor.::
import scalala.tensor.mutable._
import scalala.tensor.dense._
import scalala.library.Library._
import scalala.library.Plotting._

/**
 * Created by IntelliJ IDEA.
 * User: Shingo Omura <everpeace_at_gmail_dot_com>
 */

object MultiVariateLinearRegressionSample {

  def main(args: Array[String]): Unit = run

  def run: Unit = {
    // データの読み込み
    val reg = "([0-9]+)\\,([0-9]+)\\,([0-9]+)*".r
    val data = DenseMatrix(fromFile("MultiVariateLinearRegression.txt").getLines().toList.flatMap(_ match {
      case reg(x1, x2, y) => Seq((x1.toDouble, x2.toDouble, y.toDouble))
      case _ => Seq.empty
    }): _*)
    println("読み込まれたデータ:\n\t広さ\t#寝室\t値段\n" + data)

    // 学習データの正規化
    var X = data(::, 0 to 1)
    val y = data(::, 2)
    val norm = normalizeFeatures(X)
    print("\n\n平均(広さ, #寝室) =" + norm._2)
    print("標準偏差(広さ, #寝室) =" + norm._3)
    // バイアス項を加えて3次元に [1 ; normalized_x1 ; normalized_x2]
    X = DenseMatrix.horzcat(DenseMatrix.ones[Double](X.numRows, 1), norm._1)
    print("正規化されたデータ(バイアス項が追加されている):\n" + X)

    // 学習パラメータ
    val alpha = 0.1d // 最急降下法へのパラメータ
    val num_iters = 100 // 最急降下法の回数
    var theta = Vector.zeros[Double](3).asCol

    // 学習
    println("\n======学習開始======")
    val result = gradientDescent(X, y, theta, alpha, num_iters)
    println("======学習完了======")

    //学習結果表示
    plot((1 to num_iters).toArray, result._2.toArray)
    xlabel("number of iterations")
    ylabel("cost")
    title("History of Cost")
    print("\n学習結果(パラメータ):\t" + theta.asRow)
    println("学習結果(コスト):\t" + result._2(num_iters - 1))
    println("\n\nコスト関数履歴のウィンドウを閉じて修了してください。")
  }


  // thetaにおけるコストを算出する. ( cost = (x*theta -y )/2m)
  // X: X(i,::) == 各フィーチャー行ベクトル
  //  y: 各データに対するyの値(列ベクトル)
  // theta: パラメータ(列ベクトル)
  def computeCost(X: Matrix[Double], y: VectorCol[Double], theta: VectorCol[Double]): Double = {
    val diff = X * theta - y
    val m = y.length
    val cost = (diff.t * diff) / (2 * m)
    cost
  }

  // 最急降下法
  // 各パラメータをコストが減る方向にalpha倍してずらすのを繰り返す
  // X: X(i,::) == 各フィーチャー行ベクトル
  //  y: 各データに対するyの値(列ベクトル)
  // theta: パラメータ(列ベクトル)
  // alpha: 坂を下るときにどのくらい下るかの係数
  // num_iters: 繰り返し回数
  def gradientDescent(X: Matrix[Double], y: VectorCol[Double], theta: VectorCol[Double], alpha: Double, num_iters: Int) = {
    val m = y.length
    val p = X.numCols
    val J_history = Vector.zeros[Double](num_iters)

    for (n <- 0 until num_iters) {
      // backup previous theta
      val theta_prev = Vector.zeros[Double](theta.length).asCol
      for (i <- 0 until theta.length) theta_prev(i) = theta(i)

      // update theta
      // theta_i = theta_i - alpha* d(Cost)/d(theta_i)
      for (i <- 0 until p) {
        val derivation = ((X * theta_prev - y).t * X(::, i)) / m;
        theta(i) = (theta_prev(i) - (alpha * derivation))
      }
      J_history(n) = computeCost(X, y, theta)
      print("n=" + (n + 1).toString + "/" + num_iters + ": cost = " + J_history(n) + "\t theta = " + theta.asRow)
    }
    (theta, J_history)
  }

  // X = [ x_1; x_2; ...]の各フィーチャベクトルを平均0, 標準偏差1に正規化する
  def normalizeFeatures(X: Matrix[Double]) = {
    val mu = mean(X, Axis.Vertical)
    // 各次元でのσを求める
    val sigma = DenseVector.zeros[Double](X.numCols).asRow
    for (i <- 0 until X.numCols) sigma(i) = sqrt(variance(X(::, i)))

    // 各データに対して (x-μ)/σ を行うことで正規化
    for (i <- 0 until X.numRows)
      X(i, ::) := ((X(i, ::) :- mu) :/ sigma) //ベクトル演算によりすべての次元を同時に正規化
    (X, mu, sigma)
  }
}