package org.everpeace.scalala.sample

import scala.io.Source.fromFile
import scalala.scalar._;
import scalala.tensor.::;
import scalala.tensor.mutable._;
import scalala.tensor.dense._;
import scalala.tensor.sparse._;
import scalala.library.Library._;
import scalala.library.LinearAlgebra._;
import scalala.library.Statistics._;
import scalala.library.Plotting._;
import scalala.operators.Implicits._;


/**
 * Multi-Variate Linear Regression Sample By Scalala.
 *
 * Author: Shingo Omura <everpeace_at_gmail_dot_com>
 */

object MultiVariateLinearRegressionSample {

  def main(args: Array[String]): Unit = run

  def run: Unit = {
    // loading sample data
    val reg = "([0-9]+)\\,([0-9]+)\\,([0-9]+)*".r
    val data = DenseMatrix(fromFile("MultiVariateLinearRegression.txt").getLines().toList.flatMap(_ match {
      case reg(x1, x2, y) => Seq((x1.toDouble, x2.toDouble, y.toDouble))
      case _ => Seq.empty
    }): _*)
    println("Data Loaded:\n  Area\t#BedRooms\tPrice\n" + data)

    // normalize features
    var X = data(::, 0 to 1)
    val y = data(::, 2)
    val norm = normalizeFeatures(X)
    print("\n\naverage(Area, #BedRooms) =" + norm._2)
    print("Std. Dev.(Area, #BedRooms) =" + norm._3)
    // adding bias term ( X => [1 ; normalized_x1 ; normalized_x2] )
    X = DenseMatrix.horzcat(DenseMatrix.ones[Double](X.numRows, 1), norm._1)
    print("Normalized Data (added bias term):\n" + X)

    // learning parameters
    val alpha = 0.1d
    val num_iters = 100
    var theta = Vector.zeros[Double](3).asCol

    // optimization
    println("\n======Start Learning======")
    val result = gradientDescent(X, y, theta, alpha, num_iters)
    println("======Finish Learing======")

    // display learned result
    plot((1 to num_iters).toArray, result._2.toArray)
    xlabel("number of iterations")
    ylabel("cost")
    title("History of Cost")
    print("\nLeraned Parameters(Bias,Area,#BedRooms):\t" + theta.asRow)
    println("Learned Cost:\t" + result._2(num_iters - 1))
    println("\n\nTo finish this program, close the cost's history window.")
  }


  // calculating cost. (cost = (x*theta -y )/2m)
  // X: feature row vector is stored in each row
  // y: y's column vector coresponding to feature row vector.
  // theta: parameter column vector
  def computeCost(X: Matrix[Double], y: VectorCol[Double], theta: VectorCol[Double]): Double = {
    val diff = X * theta - y
    val m = y.length
    val cost = (diff.t * diff) / (2 * m)
    cost
  }

  // Gradient Descent
  // X: feature row vector is stored in each row
  // y: y's column vector, each value is corresponding to each feature row vector.
  // theta: parameter column vector
  // alpha: learning rate
  // num_iters: number of iterations
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

  // normalize the features to average = 0, standard deviation = 1.0
  // X: feature row vector is stored in each row.  i.e. each row indicates a specific feature.
  def normalizeFeatures(X: Matrix[Double]) = {
    // calculate μ (average)
    val mu = mean(X, Axis.Vertical)
    // calculate σ (standard deviation)
    val sigma = DenseVector.zeros[Double](X.numCols).asRow
    for (i <- 0 until X.numCols) sigma(i) = X(::, i).stddev

    // for each feature update x to (x-μ)/σ
    for (i <- 0 until X.numRows)
      X(i, ::) := (X(i, ::) :- mu) :/ sigma
    (X, mu, sigma)
  }
}