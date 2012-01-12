package org.everpeace.scalala.sample

import scala.io.Source.fromFile
import scalala.scalar._
import scalala.tensor.::
import scalala.tensor.mutable._
import scalala.tensor.dense._
import scalala.tensor.sparse._
import scalala.library.Library._
import scalala.library.LinearAlgebra._
import scalala.library.Statistics._
import scalala.library.Plotting._
import scalala.operators.Implicits._
import java.awt.{Paint, Color}

/**
 * Multi-Variate Linear Regression Sample By Scalala.
 *
 * Author: Shingo Omura <everpeace_at_gmail_dot_com>
 */

object UniVariateLinearRegressionSample {

  def main(args: Array[String]): Unit = run

  def run: Unit = {
    // loading sample data
    val reg = "(-?[0-9]*\\.[0-9]+)\\,(-?[0-9]*\\.[0-9]+)*".r
    val data: Matrix[Double] = DenseMatrix(fromFile("data/UniVariateLinearRegression.txt").getLines().toList.flatMap(_ match {
      case reg(x, y) => Seq((x.toDouble, y.toDouble))
      case _ => Seq.empty
    }): _*)

    plot.hold = true
    scatter(data(::, 0), data(::, 1), circleSize(0.3)(data.numRows), {case _ => Color.BLUE}:Int~>Paint)
    xlabel("x1")
    ylabel("x2")
    title("sample data")

    // add bias term to X
    // Scalala doesn't have DenseMatrix(VectorCol*) but DenseMatrix(VectorRow*)
    val X = DenseMatrix(DenseVector.ones[Double](data.numRows).asRow, data(::, 0).asRow).t
    val y = data(::, 1)

    // gradient descent parameters
    val alpha = 0.02d
    val num_iters = 500
    import MultiVariateLinearRegressionSample.computeCostAndGrad
    val (theta, costHist) = gradientDescent(DenseVector.zeros[Double](data.numCols), computeCostAndGrad(X, y), alpha, num_iters)

    readLine("Learning finished!  press enter to display learned function.")

    plot(X(::, 1), X * theta, colorcode = "red")
    title("sample data(in blue) and learned function(in red).")

    readLine("paused...  press enter to display cost history of learning.")
    figure(2)
    plot((1 to num_iters).toArray, costHist)
    xlabel("number of iterations")
    ylabel("cost")
    title("cost history of learning")

    println("\nTo finish this program, close all chart windows.")
  }

}