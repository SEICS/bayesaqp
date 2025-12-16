package bayestensor
package transformation

import ir._

object PartialEvaluator {
  case class InlinedFunc(f: Seq[Sym] => Exp, orig: Func) {
    def apply(args: Seq[Exp]): Exp = {
      assert(args.length == orig.ins.length)
      val nins = orig.ins.map(i => Sym.fresh(i.name))
      LetBindingN(nins.zip(args), f(nins))
    }
  }
  def inline(prog: Prog): Exp = {
    val nfuns = prog.funs.foldLeft(Seq[InlinedFunc]())((acc, cur) => {
      // val nins = cur.ins.zip(cur.ins.map(i => Sym.fresh(i.name))).toMap
      val f = (nins: Seq[Sym]) => {
        betaReduce(acc, cur.ins.zip(nins).toMap)(cur.body)
      }
      acc :+ InlinedFunc(f, cur)
    })
    betaReduce(nfuns, Map())(prog.body)
  }
  def betaReduce(funs: Seq[InlinedFunc], bindings: Map[Sym, Exp])(e: Exp): Exp = e match {
    case LetBinding(x, e1, e2) =>
      val te1 = betaReduce(funs, bindings)(e1)
      val te2 = 
        if(bindings.contains(x))
          betaReduce(funs, bindings.filterKeys(_ != x))(e2)
        else
          betaReduce(funs, bindings)(e2)
      LetBinding(x, te1, te2)
    case s @ Sym(_) =>
      bindings.getOrElse(s, s)
    case FunCall(name, args) if funs.exists(_.orig.name == name) =>
      val targs = args.map(betaReduce(funs, bindings))
      funs.find(_.orig.name == name).get.apply(targs)
    case ExpShape(cs, fact) =>
      fact(cs.map(betaReduce(funs, bindings)))
  }
  def simplify(ctx: Map[Sym, Exp])(e: Exp): Exp = e match {
    case IfThenElse(e1, e2, e3) if e2 == e3 =>
      e1 match {
        case Eq(_, _) =>
          IfThenElse(simplify(ctx)(e1), simplify(ctx)(e2), simplify(ctx)(e3))
        case _ =>
          simplify(ctx)(e2)
      }
    case LetBinding(x, LetBinding(y, e11, e12), e2) =>
      simplify(ctx)(LetBinding(y, e11, LetBinding(x, e12, e2)))
    // case LetBinding(x, y: Sym, e2) =>
    //   val c = ctx.getOrElse(y, y)
    //   simplify(ctx + (x -> c))(e2)
    // case LetBinding(x, c: Const, e2) =>
    //   simplify(ctx + (x -> c))(e2)
    // case LetBinding(x, c: Pair, e2) =>
    //   simplify(ctx + (x -> c))(e2)
    case LetBinding(x, e1, e2) =>
      val te1 = simplify(ctx)(e1)
      te1 match {
        case y: Sym => 
          val c = ctx.getOrElse(y, y)
          simplify(ctx + (x -> c))(e2)
        case c: Const =>
          simplify(ctx + (x -> c))(e2)
        case c: Pair => 
          simplify(ctx + (x -> c))(e2)
        case _ => 
          LetBinding(x, te1, simplify(ctx)(e2))
      }
    case Fst(e) =>
      simplify(ctx)(e) match {
        case Pair(e1, e2) => e1
        case te => Fst(te)
      }
    case Snd(e) =>
      simplify(ctx)(e) match {
        case Pair(e1, e2) => e2
        case te => Snd(te)
      }
    case s @ Sym(_) if ctx.contains(s) =>
      ctx(s)
    case Not(e1) =>
      simplify(ctx)(e1) match {
        case BoolLiteral(b) => BoolLiteral(!b)
        case te1 => Not(te1)
      }
    case ExpShape(cs, fact) => 
      fact(cs.map(simplify(ctx)))
  }
}