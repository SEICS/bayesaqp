package bayestensor
package ir

import scala.annotation.tailrec

case class Prog(funs: Seq[Func], body: Exp)
case class Func(name: String, ins: Seq[Sym], body: Exp)
sealed trait Exp

sealed trait Const extends Exp
case class BoolLiteral(b: Boolean) extends Const
case class IntLiteral(i: Int) extends Const
case class DoubleLiteral(d: Double) extends Const
case class VectorLiteral(v: Seq[Double]) extends Const


case class BinOp(e1: Exp, e2: Exp, op: String) extends Exp
case class UnaryOp(e: Exp, op: String) extends Exp

case class FunCall(name: String, args: Seq[Exp]) extends Exp
case class LoadDiscrete(name: String, args: Seq[Sym], dims: Seq[Exp]) extends Exp
case class LoadTensor(name: String, dims: Seq[Exp]) extends Exp
case class LoadDictionary(name: String, dims: Seq[Exp]) extends Exp
case class Sym(name: String) extends Exp
object Sym {
  var _lastid: Int = 0
  def fresh(name: String = "x"): Sym = {
    _lastid += 1
    Sym(s"${name}${_lastid}")
  }
}
case class IfThenElse(cond: Exp, thenp: Exp, elsep: Exp) extends Exp
case class LetBinding(x: Sym, e1: Exp, e2: Exp) extends Exp {
  override def hashCode(): Int = this match {
    case LetBindingN(xs, res) =>
      xs.map(xe => xe._1.hashCode() + xe._2.hashCode()).sum + res.hashCode()
    case _ => super.hashCode()
  }
}
case class MatchCase(e: Exp, cases: Seq[(Exp, Exp)]) extends Exp

case class Pair(e1: Exp, e2: Exp) extends Exp
case class Fst(e: Exp) extends Exp
case class Snd(e: Exp) extends Exp

case class Observe(e: Exp, args: Seq[Exp]) extends Exp
case class Discrete(args: Seq[Exp]) extends Exp
case class Flip(e: Exp) extends Exp

case class Select(e: Exp, args: Seq[Exp]) extends Exp
case class Weight(e1: Sym, e2: Sym) extends Exp
case class Normalize(e: Sym, args: Seq[Exp]) extends Exp // e is a distribution P, and args are list of integers representing dimonsions to normalize with
case class ToTensor(e: Sym) extends Exp
case class UnitMatrix(e: Sym) extends Exp
case class Query(vars: Seq[Sym], selections: Seq[Sym], observations: Seq[Sym], weights: Seq[Sym]) extends Exp
case class Expectation(normVars: Seq[Sym], dist: Sym, binsMean: Sym, dims: Seq[Exp]) extends Exp // dist is a distribution P, binsMean is the binsMean(P)
case class OuterProduct(e1: Sym, e2: Sym) extends Exp
case class Filter(e: Exp, args: Seq[Exp]) extends Exp
case class Lift(e: Exp) extends Exp
case class ReverseMap(e: Exp, dicts: Seq[Exp]) extends Exp
case class Decompress(e1: Exp, e2: Exp, args: Seq[Exp]) extends Exp

object ObserveNorm {
  def unapply(e: Observe): Option[(Sym, Seq[Int])] = e match {
    case Observe(x: Sym, args) if args.forall(_.isInstanceOf[IntLiteral]) =>
      Some(x -> args.collect({ case IntLiteral(i) => i }))
    case _ =>
      None
  }
}

object SelectNorm {
  def unapply(e: Select): Option[(Sym, Seq[Int])] = e match {
    case Select(x: Sym, args) if args.forall(_.isInstanceOf[IntLiteral]) =>
      Some(x -> args.collect({ case IntLiteral(i) => i }))
    case _ =>
      None
  }
}

abstract class UnaryOpFactory(op: String) {
  def apply(e: Exp) = UnaryOp(e, op)
  def unapply(e: Exp): Option[Exp] = e match {
    case UnaryOp(e1, op1) if op == op1 => Some(e1)
    case _ => None
  }
}

object Not extends UnaryOpFactory("!")
object Neg extends UnaryOpFactory("-")

abstract class BinOpFactory(op: String) {
  def apply(e1: Exp, e2: Exp) = BinOp(e1, e2, op)
  def unapply(e: Exp): Option[(Exp, Exp)] = e match {
    case BinOp(e1, e2, op1) if op == op1 => Some((e1, e2))
    case _ => None
  }
}

object Add extends BinOpFactory("+")
object Sub extends BinOpFactory("-")
object Mult extends BinOpFactory("*")
object Div extends BinOpFactory("/")
object Pow extends BinOpFactory("^")
object And extends BinOpFactory("&&")
object Or extends BinOpFactory("||")
object Imp extends BinOpFactory("~>")
object Eq extends BinOpFactory("==")

object LetBindingN {
  def apply(bindings: Seq[(Sym, Exp)], body: Exp): Exp =
    bindings.foldRight(body)((cur, acc) => LetBinding(cur._1, cur._2, acc))
  type Res = Option[(Seq[(Sym, Exp)], Exp)]
  def unapply(exp: Exp): Res =
    rec(exp, None)

  @tailrec def rec(exp: Exp, res: Res): Res = exp match {
    case LetBinding(x, e1, e2) => rec(e2, res match {
      case Some((seq, b)) => Some((seq :+ (x -> e1)) -> e2)
      case None => Some(Seq(x -> e1) -> e2)
    })
    case _ => res
  }
}

object TupleN {
  def unapply(e: Exp): Option[Seq[Exp]] = e match {
    case Pair(TupleN(s1), TupleN(s2)) => Some(s1 ++ s2)
    case e1 => Some(Seq(e1))
  }
  def apply(args: Seq[Exp]): Exp = args match {
    case Seq() => ???
    case Seq(s) => s
    case _ => Pair(args.head, TupleN(args.tail))
  }
}

object MatchVars {
  def unapply(e: Exp): Option[Seq[Sym]] = e match {
    case s @ Sym(_) => Some(Seq(s))
    case Pair(MatchVars(s1), MatchVars(s2)) => Some(s1 ++ s2)
    case _ => None
  }
  def apply(vars: Seq[Sym]): Exp = TupleN(vars)
}

object ExpShape {
  def unapply(e: Exp): Option[(Seq[Exp], Seq[Exp] => Exp)] = e match {
    case BoolLiteral(b) => Some(Seq() -> (x => BoolLiteral(b)))
    case IntLiteral(i) => Some(Seq() -> (x => IntLiteral(i)))
    case DoubleLiteral(d) => Some(Seq() -> (x => DoubleLiteral(d)))
    case VectorLiteral(v) => Some(Seq() -> (x => VectorLiteral(v)))
    case BinOp(e1, e2, op) => Some(Seq(e1, e2) -> (cs => BinOp(cs(0), cs(1), op)))
    case UnaryOp(e1, op) => Some(Seq(e1) -> (cs => UnaryOp(cs(0), op)))
    case Sym(name) => Some(Seq() -> (x => Sym(name)))
    case IfThenElse(e1, e2, e3) => Some(Seq(e1, e2, e3) -> (cs => IfThenElse(cs(0), cs(1), cs(2))))
    case LetBinding(x, e1, e2) => Some(Seq(x, e1, e2) -> (cs => LetBinding(cs(0).asInstanceOf[Sym], cs(1), cs(2))))
    case MatchCase(e1, cs1) => Some((Seq(e1) ++ cs1.map(_._2)) -> (cs => MatchCase(cs(0), cs1.map(_._1).zip(cs.tail))))
    case Pair(e1, e2) => Some(Seq(e1, e2) -> (cs => Pair(cs(0), cs(1))))
    case Fst(e1) => Some(Seq(e1) -> (cs => Fst(cs(0))))
    case Snd(e1) => Some(Seq(e1) -> (cs => Snd(cs(0))))
    case Observe(e1, args) => Some((Seq(e1) ++ args) -> (cs => Observe(cs.head, cs.tail)))
    case Select(e1, args) => Some((Seq(e1) ++ args) -> (cs => Select(cs.head, cs.tail)))
    case Filter(e1, args) => Some((Seq(e1) ++ args) -> (cs => Filter(cs.head, cs.tail)))
    case Discrete(args) => Some(args -> (cs => Discrete(cs)))
    case ToTensor(arg) => Some(Seq(arg) -> (cs => ToTensor(cs(0).asInstanceOf[Sym])))
    case UnitMatrix(arg) => Some(Seq(arg) -> (cs => UnitMatrix(cs(0).asInstanceOf[Sym])))
    case Normalize(e1, args) => Some((Seq(e1) ++ args) -> (cs => Normalize(cs.head.asInstanceOf[Sym], cs.tail)))
    case OuterProduct(e1, e2) => Some((Seq(e1, e2)) -> (cs => OuterProduct(cs(0).asInstanceOf[Sym], cs(1).asInstanceOf[Sym])))
    case Weight(e1, e2) => Some((Seq(e1, e2)) -> (cs => Weight(cs(0).asInstanceOf[Sym], cs(1).asInstanceOf[Sym])))
    case Lift(e1) => Some((Seq(e1)) -> (cs => Lift(cs(0))))
    case ReverseMap(e, dicts) => Some((Seq(e) ++ dicts) -> (cs => ReverseMap(cs.head, cs.tail)))
    case Decompress(e1, e2, args) => Some(Seq(e1, e2) ++ args -> (cs => Decompress(cs(0), cs(1), cs.takeRight(args.size))))
    case Flip(e1) => Some(Seq(e1) -> (cs => Flip(cs(0))))
    case FunCall(name, args) => Some(args -> (cs => FunCall(name, cs)))
    case LoadDiscrete(name, args, dims) => 
      Some(Seq() -> (cs => LoadDiscrete(name, args, dims)))
    case LoadTensor(name, dims) => 
      Some(Seq() -> (cs => LoadTensor(name, dims)))
    case LoadDictionary(name, dims) => 
      Some(Seq() -> (cs => LoadDictionary(name, dims)))
    case Query(vs, sels, obss, weights) => 
      // FIXME
      Some(Seq() -> (cs => Query(vs, sels, obss, weights)))
    case Expectation(vs, dist, bm, dims) => 
      // FIXME
      Some(Seq() -> (cs => Expectation(vs, dist, bm, dims)))
  }
}