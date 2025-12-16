package bayestensor
package transformation

import ir._

object CPT {
  implicit object SeqOrdering extends Ordering[Seq[Int]] {
    val ord = implicitly[Ordering[Int]]
    def compare(a: Seq[Int], b: Seq[Int]): Int = {
      val xe = a.iterator
      val ye = b.iterator
      
      while (xe.hasNext && ye.hasNext) {
        val res = ord.compare(xe.next(), ye.next())
        if (res != 0) return res
      }
      Ordering.Boolean.compare(xe.hasNext, ye.hasNext)
    }
  }

  def fromExp(e: Exp)(ctx: Map[Sym, TensorDim]): TensorExp = e match {
    case Discrete(vs) => 
      val probs = vs.map(v => v match {
        case IntLiteral(i) => i.toDouble
        case DoubleLiteral(d) => d
        case _ => raise(s"wrong discrete argument: `$v`")
      })
      TensorConst(probs, TensorDim(1, Seq(vs.length)))
    case MatchCase(MatchVars(vs), body) =>
      val tdims = vs.map(ctx.apply _)
      val sortedBodies = body.map(x => x._1 match {
        case TupleN(idxs) if idxs.forall(_.isInstanceOf[IntLiteral]) =>
          val is = idxs.map(x => x.asInstanceOf[IntLiteral].i)
          fromExp(x._2)(ctx) match {
            case TensorConst(vs, dim) =>
              is -> (vs -> dim.size)
            case bv =>
              raise(s"wrong body for match expression `$bv`. It should be `discrete`.")
          }
        case mt => raise(s"wrong match expression `$mt`")
      }).sortBy(_._1)
      if(sortedBodies.size != tdims.map(_.size).product)
        raise(s"pattern matching not exhaustive [${sortedBodies.size}]: `$body`")
      val n = sortedBodies.head._2._2
      sortedBodies.find(x => x._2._2 != n) match {
        case Some(v) => 
          raise(s"The size of output tensor should be $n but is `${v._2._2}` somewhere.")
        case None => 
      }
      TensorConst(sortedBodies.flatMap(_._2._1), TensorDim(vs.size + 1, tdims.map(_.size) ++ Seq(n)))
    case LoadDiscrete(name, args, dims) => 
      TensorLoad(name, args.map(x => TensorSym(x.name, TensorDim.Scalar)).toSeq, TensorDim(dims.length, dims.map(x => x.asInstanceOf[IntLiteral].i).toSeq))
    case _ =>
      raise(s"Cannot infer CPT for $e")     
  }
}