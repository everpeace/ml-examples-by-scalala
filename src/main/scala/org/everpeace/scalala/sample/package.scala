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
  def meshgrid(x: Vector[Double], y: Vector[Double]): (Matrix[Double], Matrix[Double]) = {
    val xMesh = DenseMatrix.zeros[Double](y.length, x.length)
    for (i <- 0 until y.length) {
      xMesh(i, ::) := x.asRow
    }
    val yMesh = DenseMatrix.zeros[Double](y.length, x.length)
    for (i <- 0 until x.length) {
      yMesh(::, i) := y.asCol
    }
    (xMesh, yMesh)
  }
}