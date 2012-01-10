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


/**
 * Support Vector Machine With Gaussian Kernel Sample By Scalala.
 *
 * Author: Shingo Omura <everpeace_at_gmail_dot_com>
 */

object SupportVectorMachineWithGaussianKernel {
  type ~>[-A, +B] = PartialFunction[A, B]

  def main(args: Array[String]): Unit = run

  def run: Unit = {
    //TODO under construction
  }

  def gaussianKernel(sigma: Double)(x1: VectorCol[Double], x2: VectorCol[Double]): Double
  = exp(-1 * ((x1 :- x2).t * (x1 - x2)) / (2 * sigma * sigma))

  def trainSVM(X: Matrix[Double], Y: VectorCol[Double], C: Double,
               kernel: (VectorCol[Double], VectorCol[Double]) => Double,
               tol: Double = 1e-3, max_passes: Int = 5): Unit = {
    val m = X.numRows
    val n = X.numCols
    val Y2 = Vector.vertcat(Y)
    Y2(Y findAll (_ == 0)) := -1 // remap 0 to -1
    val alphas = Vector.zeros[Double](m)
    var b = 0.0d
    val E = Vector.zeros[Double](m)
    var passes = 0
    var eta = 0.0d
    var L = 0.0d
    var H = 0.0d

    // generate Kernel Matrix
    val K: Matrix[Double] = DenseMatrix.zeros[Double](m, m)
    for (i <- 0 until m; j <- i until m) {
      K(i, j) = kernel(X(i, ::).t, X(j, ::).t)
      K(j, i) = K(i, j) // the matrix is symmetric.
    }

    while (passes < max_passes) {
      var num_alpha_changed = 0
      for (i <- 0 until m) {
        E(i) = b + (alphas :* (Y2 :* K(::, i))).sum - Y2(i)
        if ((Y2(i) * E(i) < -tol && alphas(i) < C) || (Y2(i) * E(i) > tol && alphas(i) > 0)) {
          var j = scala.math.ceil((m - 1) * scala.util.Random.nextDouble()).toInt
          // Make sure i \neq j
          while (j == i) (j = scala.math.ceil((m - 1) * scala.util.Random.nextDouble()).toInt)

          //Calculate Ej = f(x(j)) - y(j) using (2).
          E(j) = b + (alphas :* (Y2 :* K(::, j))).sum - Y2(j)

          //Save old alphas
          var alpha_i_old = alphas(i);
          var alpha_j_old = alphas(j);

          // Compute L and H by (10) or (11).
          if (Y2(i) == Y2(j)) {
            L = scala.math.max(0, alphas(j) + alphas(i) - C);
            H = scala.math.min(C, alphas(j) + alphas(i));
          } else {
            L = scala.math.max(0, alphas(j) - alphas(i));
            H = scala.math.min(C, C + alphas(j) - alphas(i));
          }

          //Compute eta by (14).
          eta = 2 * K(i, j) - K(i, i) - K(j, j);

          if (L != H && eta < 0) {
            //Compute and clip new value for alpha j using (12) and (15).
            alphas(j) = alphas(j) - (Y2(j) * (E(i) - E(j))) / eta;

            //Clip
            alphas(j) = scala.math.min(H, alphas(j));
            alphas(j) = scala.math.max(L, alphas(j));

            // Check if change in alpha is significant
            if (abs(alphas(j) - alpha_j_old) < tol) {
              //continue to next i.
              // replace anyway
              alphas(j) = alpha_j_old;
            } else {
              //Determine value for alpha i using (16).
              alphas(i) = alphas(i) + Y2(i) * Y2(j) * (alpha_j_old - alphas(j));

              //Compute b1 and b2 using (17) and (18) respectively.
              var b1 = b - E(i) - (Y2(i) * (alphas(i) - alpha_i_old) * K(i, j))
              -(Y2(j) * (alphas(j) - alpha_j_old) * K(i, j))
              var b2 = b - E(j) - (Y2(i) * (alphas(i) - alpha_i_old) * K(i, j))
              -(Y2(j) * (alphas(j) - alpha_j_old) * K(j, j))

              // Compute b by (19).
              if (0 < alphas(i) && alphas(i) < C) {
                b = b1
              } else if (0 < alphas(j) && alphas(j) < C) {
                b = b2
              } else {
                b = (b1 + b2) / 2.0d;
              }

              num_alpha_changed += 1
            }
          }
        }
      }

      if (num_alpha_changed == 0) {
        passes += 1
      } else {
        passes = 0
      }
    }

    val _idx = alphas.findAll(_ > 0.0d).toSeq
    val _X = X(_idx, ::)
    val _Y = Y2(_idx, ::)
    val _kernel = kernel
    val _b = b
    val _alphas = alphas(_idx)
    val _w = ((alphas :* Y2).asRow * X).asCol

  }

}