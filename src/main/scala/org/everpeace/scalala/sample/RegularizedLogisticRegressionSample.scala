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
  type ~>[-A,+B] = PartialFunction[A,B]

  def main(args: Array[String]): Unit = run

  def run: Unit = {
    // loading sample data
    val reg = "(-?[0-9]*\\.[0-9]+)\\,(-?[0-9]*\\.[0-9]+)\\,([01])*".r
    val data: Matrix[Double] = DenseMatrix(fromFile("data/RegularizedLogisticRegression.txt").getLines().toList.flatMap(_ match {
      case reg(x1, x2, y) => Seq((x1.toDouble, x2.toDouble, y.toDouble))
      case _ => Seq.empty
    }): _*)
    println("Data Loaded:\nTest1Score\tTest2Score\tResult(1=accepted/0=rejected)\n" + data)

    var y = data(::, 2)
    // Scalala cannot DenseMatrix(Cols) but DenseMatrix(Rows).
    var X = DenseMatrix(mapFeatures(data(::, 0).asRow, data(::, 1).asRow): _*).t

    // parameters to learn.
    val init_theta = DenseVector.zeros[Double](X.numCols).asCol;
    // regularized paremeter.
    val lambda = 1d;
    // gradient descent parameters
    val alpha = 5d;
    val num_iters = 500;
    val (learnedTheta, costHistory) = gradientDescent(init_theta, costFunctionAndGrad(X, y, lambda), alpha, num_iters)

    print("paused... press enter.")
    readLine()
    println("displaying leaning history of cost value.")
    plotLeraningHistory(costHistory)
    print("paused... press enter.")
    readLine()
    println("displaying sample data(blue:accepted, red:rejected) and learned decision boundary(yellow).")
    plotDecisionBoundary(data, learnedTheta)

    val prediction = sigmoid(X * learnedTheta).map(v => if (v >= 0.5) 1.0d else 0.0d)
    val corrects = (y :== prediction).map(if (_) 1.0d else 0.0d)
    println("\nTraining Accuracy:%2.2f percent".format(mean(corrects) * 100))
    println("\nTo finish this program, close all chart windows.")
  }

  // maps the two input features to quadratic features.
  // Returns a new feature sequence with more features,
  // comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
  def mapFeatures(X1: VectorRow[Double], X2: VectorRow[Double]): Seq[VectorRow[Double]]
  = {
    val degree = 6
    for (i <- 0 to degree; j <- 0 to i) yield ((X1 :^ (i - j)) :* (X2 :^ j)).asRow
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
    scatter(posx1, posx2, Vector.fill(posIdx.length)(0.03), {case _ => Color.BLUE}:Int~>Paint, tips = {case i: Int => acceptedTips(i)}:Int~>String, name = "accepted")

    val negx1 = x1(negIdx: _*)
    val negx2 = x2(negIdx: _*)
    val rejectedTip= (i:Int) => "REJECTED(" + negx1(i).toString + "," + negx2(i).toString + ")"
    scatter(negx1, negx2, Vector.fill(negIdx.length)(0.03), {case _ => Color.RED}:Int~>Paint, tips = {case i: Int => rejectedTip(i)}:Int~>String, name = "rejected")

    xlabel("Test1 score")
    ylabel("Test2 score")
  }

  // plot decision boundary
  // scalala doesn't have contour, so this searches boundary manually.
  def plotDecisionBoundary(data: Matrix[Double], theta: VectorCol[Double]): Unit = {
    figure(2)
    plot.hold = true
    plotSampleData(data)

    // the grid range
    val u = linspace(-1, 1.5, 100);
    val v = linspace(-1, 1.5, 100);
    val z = DenseMatrix.zeros[Double](u.length, v.length);

    // Evaluate accepted probability
    for (i <- 0 until u.length; j <- 0 until v.length) {
      val mappedUV = DenseMatrix(mapFeatures(DenseVector(u(i)).asRow, DenseVector(v(j)).asRow): _*).t
      val p = sigmoid(DenseVector(mappedUV(0, ::) * theta).asCol)
      z(i, j) = p(0)
    }

    // calculate boundary manually
    val boundary = for (p <- z.findAll(v => (v - 0.5).abs <= 0.01).toSeq) yield p
    val boundaryX = for (p <- boundary) yield p._1
    val boundaryY = for (p <- boundary) yield p._2

    //plot boundary
    scatter(u(boundaryX: _*), v(boundaryY: _*), Vector.fill(boundary.length)(0.03), {case _ => Color.YELLOW}:Int~>Paint, tips = {case _ => "boundary"}:Int~>String, name = "boundary")
    title("yellow circles indicate decision boundary")
    plot.hold = false
  }
}