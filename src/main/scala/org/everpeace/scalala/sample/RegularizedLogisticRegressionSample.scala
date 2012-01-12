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
import scala.PartialFunction
import java.awt.{Color, Paint}


/**
 * Regularized Logistic Regression Sample By Scalala.
 *
 * Author: Shingo Omura <everpeace_at_gmail_dot_com>
 */

object RegularizedLogisticRegressionSample {

  def main(args: Array[String]): Unit = run

  def run: Unit = {
    // loading sample data
    val reg = "(-?[0-9]*\\.[0-9]+)\\,(-?[0-9]*\\.[0-9]+)\\,([01])*".r
    val data: Matrix[Double] = DenseMatrix(fromFile("data/RegularizedLogisticRegression.txt").getLines().toList.flatMap(_ match {
      case reg(x1, x2, y) => Seq((x1.toDouble, x2.toDouble, y.toDouble))
      case _ => Seq.empty
    }): _*)
    println("Data Loaded:\nTest1Score\tTest2Score\tResult(1=accepted/0=rejected)\n" + data)

    // Scalala cannot DenseMatrix(Cols) but DenseMatrix(Rows).
    var X = mapFeatures(data(::, 0), data(::, 1))
    var y = data(::, 2)

    // parameters to learn.
    val init_theta = DenseVector.zeros[Double](X.numCols).asCol;
    // regularized paremeter.
    val lambda = 1d;
    // gradient descent parameters
    val alpha = 5d;
    val num_iters = 500;
    val (learnedTheta, costHistory)
    = gradientDescent(init_theta,
                                  costFunctionAndGrad(X, y, lambda),
                                  alpha, num_iters)
    val accr = accuracy(y, predict(learnedTheta)(data(::,0 to 1)))
    println("\nTraining Accuracy:%2.2f percent\n\n".format(accr * 100))

    print("paused... press enter to plot learning results.")
    readLine()
    println("displaying leaning history of cost value.")
    subplot(2, 1, 1)
    plotLeraningHistory(costHistory)
    println("displaying sample data(blue:accepted, red:rejected) and learned decision boundary(yellow).")
    subplot(2, 1, 2)
    plotDecisionBoundary(data, learnedTheta)

    println("\nTo finish this program, close all chart windows.")
  }

  // maps the two input features to quadratic features.
  // Returns a new feature sequence with more features,
  // comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
  def mapFeatures(X1: Vector[Double], X2: Vector[Double]): Matrix[Double]
  = {
    val degree = 6
    val featureRows: Seq[VectorRow[Double]]
    = for (i <- 0 to degree; j <- 0 to i) yield ((X1.asRow :^ (i - j)) :* (X2.asRow :^ j))
    DenseMatrix(featureRows: _*).t
  }


  // compute sigmoid functions to each value for the column vector
  // scalala cannot exp(Vector)!  (scalala doesn't define CanExp)
  def sigmoid(v: VectorCol[Double]): VectorCol[Double] = 1 :/ (1 :+ (-v).map(exp(_)))

  // compute cost and grad  for regularized logistic regression.
  def costFunctionAndGrad(X: Matrix[Double], y: VectorCol[Double], lambda: Double)(theta: VectorCol[Double]): (Double, VectorCol[Double]) = {
    assert(X.numRows == y.length)
    assert(X.numCols == theta.length)

    val h = sigmoid(X * theta)
    val m = y.length
    // calculate penalty excluded the first theta value (bias term)
    val _theta = DenseVector(theta.values.toSeq: _*).asCol
    _theta(0) = 0
    val p = lambda * ((_theta.t * _theta) / (2 * m))

    //  cost = ((-y)'* log (h) - (1 - y)'* log (1 - h)) / m +p;
    val cost = (((-y).t * h.map(log(_))) - ((1 :- y).t * ((1 :- h)).map(log(_)))) / m + p

    // calculate grads
    // grad = (X'*(h - y) + lambda * _theta) / m;
    val grad = ((X.t * (h :- y)) :+ (lambda :* _theta)) / m

    (cost, grad)
  }

  // predict whether each sample is accepted or not.
  def predict(theta: Vector[Double])(X: Matrix[Double]): Vector[Double] = {
    val mapped = mapFeatures(X(::, 0), X(::, 1))
    sigmoid(mapped * theta.asCol).map(p => if (p >= 0.5) 1.0d else 0.0d)
  }

  // plot History of cost
  def plotLeraningHistory(cost_hist: VectorCol[Double]): Unit = {
    figure(1)
    plot((1 to cost_hist.length).toArray, cost_hist)
    xlabel("number of iterations")
    ylabel("cost")
    title("learning cost history")
  }

  // plot samples
  def plotSampleData(data: Matrix[Double]): Unit = {
    val posIdx = data(::, 2).findAll(_ == 1.0).toSeq
    val negIdx = data(::, 2).findAll(_ == 0.0).toSeq
    val x1 = data(::, 0)
    val x2 = data(::, 1)

    val posx1 = x1(posIdx: _*)
    val posx2 = x2(posIdx: _*)
    val acceptedTips = (i: Int) => "ACCEPTED(" + posx1(i).toString + "," + posx2(i).toString + ")"
    scatter(posx1, posx2, circleSize(0.03)(posIdx.length), {
      case _ => Color.BLUE
    }: Int ~> Paint,
      tips = {
        case i: Int => acceptedTips(i)
      }: Int ~> String, name = "accepted")

    val negx1 = x1(negIdx: _*)
    val negx2 = x2(negIdx: _*)
    val rejectedTip = (i: Int) => "REJECTED(" + negx1(i).toString + "," + negx2(i).toString + ")"
    scatter(negx1, negx2, circleSize(0.03)(negIdx.length), {
      case _ => Color.RED
    }: Int ~> Paint,
      tips = {
        case i: Int => rejectedTip(i)
      }: Int ~> String, name = "rejected")

    xlabel("Test1 score")
    ylabel("Test2 score")
  }

  // plot decision boundary
  // scalala doesn't have contour, so this searches boundary manually.
  def plotDecisionBoundary(data: Matrix[Double], theta: VectorCol[Double]): Unit = {
    plot.hold = true
    plotSampleData(data)

    // compute decision boundaries
    val x1range = linspace(-1, 1.5, 100);
    val x2range = linspace(-1, 1.5, 100);
    val (bx, by) = computeDecisionBoundary(x1range, x2range, predict(theta))

    //plot boundary
    scatter(bx, by, circleSize(0.03)(bx.length), {
      case _ => Color.YELLOW
    }: Int ~> Paint)
    title("Learned Decision Boundary\n (blue:accepted, red:rejected, yellow: boundary)")
    plot.hold = false
  }
}