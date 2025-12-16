package bayestensor

import scala.language.implicitConversions
import scala.annotation.implicitNotFound

/**
 * A basic pretty-printing library that handles indentation and conditional line-breaking.
 *
 * The most convenient way to build a Document is with a combination of `doc"..."` string interpolation.
 * To use `doc` string interpolation, you have to import `utils._`.
 *
 */
package object utils {
  import Document._

  /**
   * Describes how to lift objects using the `doc` string interpolation.
   */
  @implicitNotFound("Cannot find document.Liftable implementation for type ${T}")
  trait Liftable[-T] {
    def apply(value: T): Document
  }

  object Liftable {

    def apply[T](f: T => Document): Liftable[T] =
      new Liftable[T] { def apply(value: T): Document = f(value) }

  }

  implicit def toDocumentContext(ctx: StringContext): DocumentContext = new DocumentContext(ctx)

  implicit def toDocument[T: Liftable](x: T): Document = implicitly[Liftable[T]].apply(x)

  implicit val docLift: Liftable[Document] = Liftable[Document](identity)

  implicit val intLift: Liftable[Int] = Liftable[Int](_.toString)

  implicit val strLift: Liftable[String] = Liftable[String](text)

  // /** Lifting document lists with line breaks in between is almost never what the user wants, and leads to errors */
  //  implicit def seqLift[T: Liftable]: Liftable[Seq[T]] = Liftable[Seq[T]] { ls =>
  //    val lift = implicitly[Liftable[T]].apply _
  //    (ls map lift).mkDocument()
  //  }

  implicit class ListDocumentOps(docs: Seq[Document]) {
    def mkDocument(pre: Document, sep: Document, post: Document): Document = docs filter (_ != empty) match {
      case d +: ds => pre :: (d +: ds.map(sep :: _)).foldLeft(empty)(_ :: _) :: post
      case _       => pre :: post
    }
    def mkDocument(sep: Document = empty): Document = mkDocument(empty, sep, empty)
  }

  def time[A](a: => A, msg: String) = {
    val start = System.nanoTime
    val result = a
    val end = (System.nanoTime - start) / (1000 * 1000)
    System.out.println(s"$msg completed in ${Console.YELLOW}$end${Console.RESET} milliseconds")
    result
  }

  def timeFunc[I, O](msg: String)(f: I => O): I => O = { (i: I) =>
    time(f(i), msg)
  }

  implicit final class AnyOps[A](private val self: A) extends AnyVal {
    @inline def |>[B](rhs: A => B): B = rhs(self)
  }

  implicit final class FuncOps[A, B](private val self: A => B) extends AnyVal {
    @inline def >>[C](rhs: B => C): A => C = self andThen rhs
  }

  var DEBUG_FLAG: Int = -1

  def dprint(l: Int)(a: Any): Unit = 
    if(l <= DEBUG_FLAG)
      print(a)

  def dprintln(l: Int)(a: Any): Unit = 
    if(l <= DEBUG_FLAG)
      println(a)

  type Dag[T] = Map[T, Seq[T]]

  implicit class DagOps[T](dag: Dag[T]) {
    def addEdges(src: T, dsts: Seq[T]): Dag[T] = {
      dag + (src -> (dag.getOrElse(src, Seq()) ++ dsts)) // If src doesn't exist, create a edge to add destinations to existing DAG graph.
    }
    def reverse: Dag[T] = {
      dag.toSeq.flatMap(kv => kv._2.map(vx => vx -> kv._1)).groupBy(_._1).mapValues(_.map(_._2)).toMap
    }
  }

  // type Tensor[T] = Seq[T]
  // implicit class DagOps[T](dag: Dag[T]) {
  //   def add(t1: Seq[T], t2: T): Tensor[T] = {
  //     t1 + ()
  //   }
  // }
}
