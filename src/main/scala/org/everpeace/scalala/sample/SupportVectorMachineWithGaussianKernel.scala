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
import java.awt.{Color, Paint}


/**
 * Support Vector Machine With Gaussian Kernel Sample By Scalala.
 *
 * Author: Shingo Omura <everpeace_at_gmail_dot_com>
 */

object SupportVectorMachineWithGaussianKernel {

  def main(args: Array[String]): Unit = run

  def run: Unit = {
    // loading sample data
    val reg = "(-?[0-9]*\\.[0-9]+)\\,(-?[0-9]*\\.[0-9]+)\\,([01])*".r
    val data: Matrix[Double] = DenseMatrix(fromFile("data/SupportVectorMachineWithGaussianKernel.txt").getLines().toList.flatMap(_ match {
      case reg(x1, x2, y) => Seq((x1.toDouble, x2.toDouble, y.toDouble))
      case _ => Seq.empty
    }): _*)
    println("Data Loaded:\nX-value\tY-value\tResult(1=accepted/0=rejected)\n" + data)

    // plot sample
    var X = data(::, 0 to 1)
    var y = data(::, 2)
    scatter(X(::, 0), X(::, 1), circleSize(0.01)(X.numRows), y2Color(y))
    xlabel("X-value")
    ylabel("Y-value")
    title("Input data")

    // learning parameter
    // C: regularized parameter
    // sigma: gaussian Kernel parameter
    val C = 1d
    val sigma = 0.1d

    // learn svm
    println("\n\npaused... press enter to start learning SVM.")
    readLine
    val model = trainSVM(X, y, C, gaussianKernel(sigma))
    val accr = accuracy(y, predict(model)(X))
    println("\nTraining Accuracy:%2.2f percent\n\n".format(accr * 100))

    // plotting decision boundary
    println("paused... press enter to plot leaning result.")
    readLine
    plotDecisionBoundary(X, y, model)

    println("\n\nTo finish this program, close the result window.")
  }

  // gaussian kernel
  def gaussianKernel(sigma: Double)(x1: Vector[Double], x2: Vector[Double]): Double
  = {
    val _x1 = x1.asCol
    val _x2 = x2.asCol
    exp(-1 * ((_x1 - _x2).t * (_x1 - _x2)) / (2 * sigma * sigma))
  }

  // SVM Model
  case class Model(X: Matrix[Double], y: Vector[Double], kernelF: (Vector[Double], Vector[Double]) => Double,
                   b: Double, alphas: Vector[Double], w: Vector[Double])

  // predict by SVM Model
  def predict(model: Model)(X: Matrix[Double]): Vector[Double] = {
    val pred = Vector.zeros[Double](X.numRows)
    val p = Vector.zeros[Double](X.numRows)
    for (i <- 0 until X.numRows) {
      var prediction = 0d;
      for (j <- 0 until model.X.numRows) {
        prediction = prediction + model.alphas(j) * model.y(j) * model.kernelF(X(i, ::), model.X(j, ::))
      }
      p(i) = prediction + model.b;
    }
    pred(p.findAll(_ >= 0)) := 1.0d
    pred(p.findAll(_ < 0)) := 0.0d
    pred
  }

  // train SVM
  // This is a simplified version of the SMO algorithm for training SVMs.
  def trainSVM(X: Matrix[Double], Y: VectorCol[Double], C: Double,
               kernel: (Vector[Double], Vector[Double]) => Double,
               tol: Double = 1e-3, max_passes: Int = 5): Model = {
    val m = X.numRows
    val n = X.numCols
    val Y2 = Vector.vertcat(Y)
    Y2(Y findAll (_ == 0d)) := -1d // remap 0 to -1
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

    print("Training(C=%f) (This takes a few minutes.)\n".format(C))
    var dots = 0
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

      print(".")
      dots += 1
      if (dots > 78) {
        print("\n")
        dots = 0
      }
    }
    print("Done! \n\n")

    val _idx = alphas.findAll(_ > 0.0d).toSeq
    val _X = X(_idx, ::)
    val _Y = Y2(_idx)
    val _kernel = kernel
    val _b = b
    val _alphas = alphas(_idx)
    val _w = ((alphas :* Y2).asRow * X).asCol

    Model(_X, _Y, _kernel, _b, _alphas, _w)
  }

  def plotDecisionBoundary(X: Matrix[Double], y: Vector[Double], model: Model) = {
    print("Detecting decision boundaries...")
    // compute decision boundary.
    val NUM = 100
    val x1 = linspace(X(::, 0).min, X(::, 0).max, NUM)
    val x2 = linspace(X(::, 1).min, X(::, 1).max, NUM)
    val (bx1, bx2) = computeDecisionBoundary(x1, x2, predict(model))
    print(" Done!\n")

    // plot input data and detected boundary
    clf
    plot.hold = true
    scatter(X(::, 0), X(::, 1), circleSize(0.01)(X.numRows), y2Color(y))
    scatter(bx1, bx2, circleSize(0.01)(bx1.size), {
      case i => Color.YELLOW
    }: Int ~> Paint)
    xlabel("X-value")
    ylabel("Y-value")
    title("Learning result by SVM\n blue:accepted, red: rejected, yellow:learned decision boundary")
  }

  val i2color: Int => Paint = _ match {
    case 1 => Color.BLUE //accepted
    case 0 => Color.RED //rejected
    case _ => Color.BLACK //other
  }
  val y2Color: Vector[Double] => (Int ~> Paint) = y => {
    case i => i2color(y(i).toInt)
  }
}