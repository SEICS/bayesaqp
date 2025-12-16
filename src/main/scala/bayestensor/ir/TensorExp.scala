package bayestensor
package ir
import scala.annotation.tailrec

sealed trait TensorExp {
  def dim: TensorDim
}

case class TensorConst(elems: Seq[Double], dim: TensorDim) extends TensorExp
case class TensorContract(args: Seq[TensorExp], einsum: String, dim: TensorDim) extends TensorExp
case class TensorSym(name: String, dim: TensorDim) extends TensorExp
case class TensorLet(sym: TensorSym, e1: TensorExp, e2: TensorExp) extends TensorExp {
  def dim: TensorDim = e2.dim
}

case class TensorOnes(name: String, dim: TensorDim) extends TensorExp // like Python numpy.ones
// TODO TensorLoad: args can be removed 
case class TensorLoad(name: String, args: Seq[TensorSym], dim: TensorDim) extends TensorExp 
case class DictionaryLoad(name: String, dim: TensorDim) extends TensorExp
case class TensorLift(e: TensorExp) extends TensorExp {
  def dim: TensorDim = e.dim
}
case class TensorReverseMap(e: TensorExp, dicts: Seq[TensorExp], dim: TensorDim) extends TensorExp 
case class TensorArithmetic(e1: TensorSym, e2: TensorSym, op: TensorMathOp) extends TensorExp {
  def dim: TensorDim = e1.dim
}

case class TensorMathOp(opStr: String) extends TensorExp {
  def dim: TensorDim = TensorDim.Scalar // TOBEFIXED: wrong. operator should not have dimension
}

object TensorMathOp {
  val Multiply = TensorMathOp("*")
  val Subtract = TensorMathOp("-")
  val Add = TensorMathOp("+")
  val Divide = TensorMathOp("/")
}

case class TensorBinOp(e1: TensorExp, e2: TensorExp, op: String) extends TensorExp {
  def dim: TensorDim = TensorDim.Scalar
}
// case class UnaryOp(e: TensorExp, op: String) extends TensorExp
case class TensorFunCall(name: String, args: Seq[TensorExp]) extends TensorExp {
  def dim: TensorDim = TensorDim.Scalar
}

abstract class TensorBinOpFactory(op: String) {
  def apply(e1: TensorExp, e2: TensorExp) = TensorBinOp(e1, e2, op)
  // def dim: TensorDim = TensorDim.Scalar
  def unapply(e: TensorExp): Option[(TensorExp, TensorExp)] = e match {
    case TensorBinOp(e1, e2, op1) if op == op1 => Some((e1, e2))
    case _ => None
  }
}

object TensorAdd extends TensorBinOpFactory("+")
object TensorSub extends TensorBinOpFactory("-")
object TensorMult extends TensorBinOpFactory("*")
object TensorDiv extends TensorBinOpFactory("/")
object TensorPow extends TensorBinOpFactory("^")
object TensorAnd extends TensorBinOpFactory("&&")
object TensorOr extends TensorBinOpFactory("||")
object TensorImp extends TensorBinOpFactory("~>")
object TensorEq extends TensorBinOpFactory("==")

case class TensorSelect(e: TensorExp, filter: Seq[Int], filterDim:Int) extends TensorExp 
{
  def actualFilterDim: Int = if(filterDim == -1) e.dim.order - 1 else filterDim
  def dim: TensorDim = {
    val d = e.dim
    val fd = actualFilterDim
    TensorDim(d.order, d.dims.zipWithIndex.map(di => if(di._2 == fd) filter.size else di._1))
  }
}
case class TensorNormalize(e: TensorExp, normDims: Seq[Int]) extends TensorExp {
  def dim: TensorDim = e.dim
}

case class TensorDim(order: Int, dims: Seq[Int]) {
  def size: Int = dims.last
}

object TensorDim {
  val Scalar = TensorDim(0, Seq())
  def Vector(n: Int) = TensorDim(1, Seq(n))
}

object TensorLetN {
  def apply(bindings: Seq[(TensorSym, TensorExp)], body: TensorExp): TensorExp =
    bindings.foldRight(body)((cur, acc) => TensorLet(cur._1, cur._2, acc))
  type Res = Option[(Seq[(TensorSym, TensorExp)], TensorExp)]
  def unapply(exp: TensorExp): Res =
    rec(exp, None)

  @tailrec def rec(exp: TensorExp, res: Res): Res = exp match {
    case TensorLet(x, e1, e2) => rec(e2, res match {
      case Some((seq, b)) => Some((seq :+ (x -> e1)) -> e2)
      case None => Some(Seq(x -> e1) -> e2)
    })
    case _ => res
  }
}