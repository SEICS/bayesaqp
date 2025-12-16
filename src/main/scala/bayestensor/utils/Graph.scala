package bayestensor
package utils

object Graph {

  /* Adopted from https://gist.github.com/ThiporKong/4399695 */
  def tsort[A](edges: Traversable[(A, A)]): Iterable[A] = {
    import scala.collection.immutable.Set
    @scala.annotation.tailrec
    def tsort(toPreds: Map[A, Set[A]], done: Iterable[A]): Iterable[A] = {
      val (noPreds, hasPreds) = toPreds.partition { _._2.isEmpty }
      if (noPreds.isEmpty) {
        if (hasPreds.isEmpty) done else sys.error(hasPreds.toString)
      } else {
        val found = noPreds.map { _._1 }
        tsort(hasPreds.mapValues { _ -- found }, done ++ found)
      }
    }

    val toPred = edges.foldLeft(Map[A, Set[A]]()) { (acc, e) =>
      acc + (e._1 -> acc.getOrElse(e._1, Set())) + (e._2 -> (acc.getOrElse(e._2, Set()) + e._1))
    }
    tsort(toPred, Seq())
  }

  def schedule[T](graph: List[T], depFunction: (T, T) => Boolean): List[T] = {
    scc(graph, (x: T) => graph.filter(x2 => depFunction(x, x2))).flatten
  }

  /* Adopted from https://github.com/TiarkRompf/virtualization-lms-core/blob/develop/src/util/GraphUtil.scala */
  import java.util.{ ArrayDeque, HashMap }

  class Ref[T](init: T) {
    var value: T = init
  }

  def scc[T](start: List[T], succ: T => List[T]): List[List[T]] = {

    val id: Ref[Int] = new Ref(0)
    val stack = new ArrayDeque[T]
    val mark = new HashMap[T, Int]

    val res = new Ref[List[List[T]]](Nil)
    for (node <- start)
      visit(node, succ, id, stack, mark, res)

    res.value
  }

  def visit[T](node: T, succ: T => List[T], id: Ref[Int], stack: ArrayDeque[T],
               mark: HashMap[T, Int], res: Ref[List[List[T]]]): Int = {

    if (mark.containsKey(node))
      mark.get(node)
    else {
      id.value = id.value + 1

      mark.put(node, id.value)
      stack.addFirst(node)
      //    println("push " + node)

      var min: Int = id.value
      for (child <- succ(node)) {
        val m = visit(child, succ, id, stack, mark, res)

        if (m < min)
          min = m
      }

      if (min == mark.get(node)) {
        var scc: List[T] = Nil
        var loop: Boolean = true
        do {
          val element = stack.removeFirst()
          //        println("appending " + element)
          scc ::= element
          mark.put(element, Integer.MAX_VALUE)
          loop = element != node
        } while (loop)
        res.value ::= scc
      }
      min
    }
  }
}