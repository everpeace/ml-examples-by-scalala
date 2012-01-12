package org.everpeace.scalala

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
import scalala.generic.collection.CanMapValues

/**
 *
 * @author everpeace _at_ gmail _dot_ com
 * @date 11/12/27
 */

package object sample {
  type ~>[-A, +B] = PartialFunction[A, B]
  val circleSize = (s: Double) => (n: Int) => DenseVector.fill(n)(s)

  // gradient descent. this returns optimal theta(paremeter column vector) and history of cost value.
  // initTheta: paremeters column vector
  // func: function whose argument theta(parameter column vector) and return values are cost value and gradient vector at theta.
  // alpha: learning rate
  // num_iters: number of iterations
  def gradientDescent(initTheta: VectorCol[Double], func: (VectorCol[Double]) => (Double, VectorCol[Double]), alpha: Double, num_iters: Int): (VectorCol[Double], VectorCol[Double]) = {
    println("=== start gradientDescent loop ===")
    //initialize theta
    val theta = DenseVector.zeros[Double](initTheta.length)
    for (i <- 0 until theta.length) theta(i) = initTheta(i)
    val costHist = DenseVector.zeros[Double](num_iters)

    for (n <- 0 until num_iters) {
      print((n + 1) + "/" + num_iters + " : ")
      val r = func(theta)
      costHist(n) = r._1
      print("cost = " + r._1 + "  theta = " + theta.asRow)
      theta :-= (alpha :* r._2)
    }
    // print("RESULT: cost = " + r._1 + "  theta = " + theta.asRow)
    println("=== finish gradientDescent loop ===")
    (theta, costHist)
  }

  // construct mesh grid.
  def meshgrid(x1: Vector[Double], x2: Vector[Double]): (Matrix[Double], Matrix[Double]) = {
    val x1Mesh = DenseMatrix.zeros[Double](x2.length, x1.length)
    for (i <- 0 until x2.length) {
      x1Mesh(i, ::) := x1.asRow
    }
    val x2Mesh = DenseMatrix.zeros[Double](x2.length, x1.length)
    for (i <- 0 until x1.length) {
      x2Mesh(::, i) := x2.asCol
    }
    (x1Mesh, x2Mesh)
  }

  def computeDecisionBoundary(x1: Vector[Double], x2: Vector[Double], predict: Matrix[Double] => Vector[Double]): (Vector[Double], Vector[Double]) = {
    val (x1Mesh, x2Mesh) = meshgrid(x1, x2)
    val decisions = DenseMatrix.zeros[Double](x1Mesh.numRows, x1Mesh.numCols)

    // compute decisions for all mesh points.
    for (i <- 0 until x1Mesh.numCols) {
      val this_X: Matrix[Double] = DenseMatrix(x1Mesh(::, i).asRow, x2Mesh(::, i).asRow).t
      decisions(::, i) := predict(this_X)
    }

    // detect boundary.
    var bx1 = Seq[Double]()
    var bx2 = Seq[Double]()
    for (i <- 1 until decisions.numRows - 1; j <- 1 until decisions.numCols - 1) {
      if (decisions(i, j) == 0d && (decisions(i - 1, j - 1) == 1d || decisions(i - 1, j) == 1d || decisions(i - 1, j + 1) == 1d
        || decisions(i, j - 1) == 1d || decisions(i, j + 1) == 1d
        || decisions(i + 1, j) == 1d || decisions(i + 1, j) == 1d || decisions(i + 1, j + 1) == 1d)) {
        bx1 = x1Mesh(i, j) +: bx1
        bx2 = x2Mesh(i, j) +: bx2
      }
    }

    (Vector(bx1: _*), Vector(bx2: _*))
  }

  def accuracy(y: Vector[Double], pred: Vector[Double]): Double
  = mean((y :== pred).map(if (_) 1.0d else 0.0))
}