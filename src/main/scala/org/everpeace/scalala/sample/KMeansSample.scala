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
 * K-Means Sample By Scalala.
 *
 * Author: Shingo Omura <everpeace_at_gmail_dot_com>
 */

object KMeansSample {
  def main(args: Array[String]): Unit = run

  def run: Unit = {

    // loading sample data
    val reg = "(-?[0-9]*\\.[0-9]+)\\,(-?[0-9]*\\.[0-9]+)*".r
    val data: Matrix[Double] = DenseMatrix(fromFile("data/KMeans.txt").getLines().toList.flatMap(_ match {
      case reg(x1, x2) => Seq((x1.toDouble, x2.toDouble))
      case _ => Seq.empty
    }): _*)

    val init_centroids = DenseMatrix((3d, 3d), (6d, 2d), (8d, 5d))
    val max_iters = 10
    val (_, centroids_hist) = runKMeans(data, init_centroids, max_iters)

    println("\n\nLEARNED CENTROIDS:\n" + centroids_hist.last)

    // TODO  draw K-Mean result.
  }

  // compute each centroids.
  // X's row is each data point.
  // idx(.) is cluster label in 1..K
  // returns a matrix C s.t. C(i,::)(i=0..K-1) is the center of cluster (i+1).
  def computeCentroids(X: Matrix[Double], idx: Vector[Int], K: Int): Matrix[Double] = {
    val centroids = DenseMatrix.zeros[Double](K, X.numCols)
    for (k <- 0 until K) {
      val cluster_k = X(idx.findAll(_ == (k + 1)).toSeq, ::)
      centroids(k, ::) := (sum(cluster_k, Axis.Horizontal) / cluster_k.numRows)
    }
    centroids
  }

  // find closest centroids
  // X's row is each data point.
  // centroids' row is each center of cluster
  // returns a vector idx s.t. idx(i) is a label of cluster(1..K) of X(i,::)
  def findClosestCentroids(X: Matrix[Double], centroids: Matrix[Double]): Vector[Int] = {
    val K = centroids.numRows
    val idx = DenseVector.zeros[Int](X.numRows)
    for (i <- 0 until X.numRows) {
      val distances = Vector.zeros[Double](K)
      for (j <- 0 until K) {
        val diff: Vector[Double] = X(i, ::) - centroids(j, ::)
        distances(j) = diff.dot(diff)
      }
      idx(i) = (distances.argmin + 1)
    }
    idx
  }

  // run K-means iteratively.
  // returns history of index vectors and centroids.
  def runKMeans(X: Matrix[Double], init_centroids: Matrix[Double], max_iters: Int): (Seq[Vector[Int]], Seq[Matrix[Double]]) = {
    val K = init_centroids.numRows
    var centroids_hist = Seq[Matrix[Double]]()
    var idx_hist = Seq[Vector[Int]]()
    println("=== start K-Means loop ===")
    var centroids = init_centroids
    for (i <- 1 to max_iters) {
      println("%d/%d : ".format(i, max_iters))
      val idx = findClosestCentroids(X, centroids)
      println("idx: " + idx.asRow)
      centroids = computeCentroids(X, idx, K)
      println("centroids:\n" + centroids + "\n")
      centroids_hist = centroids +: centroids_hist
      idx_hist = idx +: idx_hist
    }
    println("=== finish K-Means loop ===")
    (idx_hist.reverse, centroids_hist.reverse)
  }
}