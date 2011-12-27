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
;


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
    val data: Matrix[Double] = DenseMatrix(fromFile("RegularizedLogisticRegression.txt").getLines().toList.flatMap(_ match {
      case reg(x1, x2, y) => Seq((x1.toDouble, x2.toDouble, y.toDouble))
      case _ => Seq.empty
    }): _*)
    println("Data Loaded:\nTest1Score\tTest2Score\tResult(1=accepted/0=rejected)\n" + data)
    plotMyData(data)

    var y = data(::, 2)
    // Scalala cannot DenseMatrix(Cols) but DenseMatrix(Rows).
    var X = DenseMatrix(mapFeatures(data(::, 0).asRow, data(::, 1).asRow, 6): _*).t

    // parameters to learn.
    val init_theta = DenseVector.zeros[Double](X.numCols).asCol;
    // regularized paremeter.
    val lambda = 155d;

    val (larnedTheta, learnedJ) = gradientDescent(init_theta, costFunctionAndGrad(X, y, lambda))
  }

  // maps the two input features to quadratic features.
  // Returns a new feature sequence with more features,
  // comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
  def mapFeatures(X1: VectorRow[Double], X2: VectorRow[Double], degree: Int): Seq[VectorRow[Double]]
  = for (i <- 0 to degree; j <- 0 to i) yield ((X1 :^ (i - j)) :* (X2 :^ j)).asRow


  // compute sigmoid functions to each value for the column vector
  // scalala cannot exp(Vector)!  (scalala doesn't define CanExp)
  def sigmoid(v: VectorCol[Double]) = 1 :/ (1 :+ (-1 :* v).map(exp(_)))

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

    //    J = ((-y) '* log (h) - (1 - y) '* log (1 - h)) / m +p;
    val J = (((-y).t * h.map(log(_))) - ((1 :- y).t * ((1 :- h)).map(log(_)))) / m + p

    // calculate grads
    // grad = (X '*(h - y) + lambda * theta1) / m;
    val grad = (X.t * (h :- y) + _theta * lambda) / m

    (J, grad)
  }

  def gradientDescent(initTheta: VectorCol[Double], func: (VectorCol[Double]) => (Double, VectorCol[Double])): (VectorCol[Double], Double) = ???

  def plotMyData(data: Matrix[Double]): Unit = {
    val posIdx = data(::, 2).findAll(_ == 1.0).toSeq
    val negIdx = data(::, 2).findAll(_ == 0.0).toSeq
    figure(1)
    plot.hold = true
    val x1 = data(::, 0)
    val x2 = data(::, 1)
    val posx1 = x1(posIdx: _*)
    val posx2 = x2(posIdx: _*)
    val acceptedColors: PartialFunction[Int, Paint] = {
      case i => Color.BLUE
    }
    val acceptedTips: PartialFunction[Int, String] = {
      case i: Int => "ACCEPTED(" + posx1(i).toString + "," + posx2(i).toString + ")"
    }
    scatter(posx1, posx2, Vector.fill(posIdx.length)(0.03), acceptedColors, tips = acceptedTips, name = "accepted")

    val negx1 = x1(negIdx: _*)
    val negx2 = x2(negIdx: _*)
    val rejectedColors: PartialFunction[Int, Paint] = {
      case i => Color.RED
    }
    val rejectedTips: PartialFunction[Int, String] = {
      case i: Int => "REJECTED(" + negx1(i).toString + "," + negx2(i).toString + ")"
    }
    scatter(negx1, negx2, Vector.fill(negIdx.length)(0.03), rejectedColors, tips = rejectedTips, name = "rejected")

    xlabel("Test1 score")
    ylabel("Test2 score")
    plot.hold = false
  }
}